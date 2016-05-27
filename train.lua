cmd = torch.CmdLine();
cmd:text('Train a image-caption ranking model');
cmd:text('Options')
cmd:option('-data','../dataset/mscoco-imcap/dataset_train.t7','Dataset for training');
cmd:text();
cmd:option('-nhword',256,'Word embedding size');
cmd:option('-nhtime',64,'Word location embedding size');
cmd:option('-ntimes',6,'Number of controller iterations');
cmd:option('-nh',512,'RNN size');
cmd:option('-nlayers',1,'RNN layers');
cmd:text();
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM)');
cmd:option('-lr',1e-3,'Learning rate');
cmd:option('-epochs',200,'Epochs. 1 epoch=1 pass of all images, rather than 1 pass of all captions');
cmd:option('-decay',100,'After decay epochs lr reduces to 0.1*lr');
params=cmd:parse(arg);
--print(params)

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim_updates'
MemNN=require('word_MemNN');

print('Initializing session');
paths.mkdir('sessions')
Session=require('session_manager');
session=Session:init('./sessions');
basedir=session:new(params);
paths.mkdir(paths.concat(basedir,'model'));
log_file=paths.concat(basedir,string.format('log.txt',1));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end

Log('Loading dataset');
dataset=torch.load(params.data);
vocabulary_size=table.getn(dataset['caption_dictionary']);
nhimage=dataset['image_fvs']:size(2);
ncaptions_per_im=dataset['ncaptions_per_im'];
nhsent=params.nh*params.nlayers*2; --Using both cell and hidden of LSTM
collectgarbage();

--Network definitions
Log('Initializing models');
--Compress sentences into memory
function embed_time(n,voc_size,nhword,nhtime)
	local w=nn.Identity()();
	local t=nn.Identity()();
	local we=nn.Dropout(0.5)(nn.LookupTable(voc_size,nhword)(w));
	local te=nn.Normalize(2)(nn.LookupTable(n,nhtime)(t));
	local output=nn.MM(true,false)({nn.View(-1,n,nhtime)(te),nn.View(-1,n,nhword)(we)});
	return nn.gModule({w,t},{output});
end
--Compute NxM matching scores of N captions and M images
function AxB(nhA,nhB)
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Dropout(0.3)(q);
	local ic=nn.Linear(nhB,nhA)(nn.Dropout(0.3)(nn.Normalize(2)(i)));
	local output=nn.MM(false,true)({qc,ic});
	return nn.gModule({q,i},{output});
end
--Create clean copies for networks
function wrap_net(net,gpu)
	local d={};
	if gpu then
		d.net=net:cuda();
	else
		d.net=net;
	end
	d.w,d.dw=d.net:getParameters();
	d.w:uniform(-0.08,0.08);
	d.deploy=d.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	return d;
end
caption_embedding_net=wrap_net(embed_time(dataset['caption_tokens']:size(2),vocabulary_size+1,params.nhword,params.nhtime),true);
caption_encoder_net=MemNN:new(MemNN.unit.lstm(params.nhword,params.nh,params.nhtime,params.nlayers,0.5),params.ntimes,params.nhtime,params.nhword,true);
multimodal_net=wrap_net(AxB(nhsent,nhimage),true);

--Criterion
criterion_1=nn.CrossEntropyCriterion():cuda();
criterion_2=nn.CrossEntropyCriterion():cuda();

--Create dummy gradients
dummy_state=torch.DoubleTensor(nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params.nhtime):fill(0):cuda();

--Optimization
Log('Setting up optimization');
niter_per_epoch=math.ceil(#dataset['ims']/params.batch);
max_iter=params.epochs*niter_per_epoch;
Log(string.format('%d iter per epoch.',niter_per_epoch));
function opt(lr,decay)
	local o={};
	o.learningRate=lr;
	o.decay=decay;
	return o;
end
decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
opt_encoder=opt(params.lr,decay);
opt_embedding=opt(params.lr,decay);
opt_multimodal=opt(params.lr,decay);

--Batch function
--Sample one caption from each of N unique images
function dataset:next_batch_train(batch_size)
	local timer = torch.Timer();
	local ims=dataset['image_fvs']:size(1);
	local iminds=torch.LongTensor(batch_size):fill(0);
	local capinds=torch.LongTensor(batch_size):fill(0);
	local labels=torch.LongTensor(batch_size):fill(0);
	iminds=torch.randperm(ims)[{{1,batch_size}}]:long();
	for i=1,batch_size do
		capinds[i]=(iminds[i]-1)*ncaptions_per_im+torch.random(ncaptions_per_im);
		labels[i]=i;
	end
	local fv_cap=dataset['caption_tokens']:index(1,capinds)+1;
	local fv_t=torch.repeatTensor(torch.range(1,dataset['caption_tokens']:size(2)):long(),batch_size,1):reshape(batch_size*dataset['caption_tokens']:size(2));
	local fv_im=dataset['image_fvs']:index(1,iminds);
	return fv_cap,fv_t,fv_im:cuda(),labels:cuda();
end

--Objective function
running_avg=0;
function ForwardBackward()
	local timer = torch.Timer();
	--clear gradients--
	caption_embedding_net.dw:zero();
	caption_encoder_net.dw:zero();
	multimodal_net.dw:zero();
	--Grab a batch--
	local fv_cap,fv_t,fv_im,labels=dataset:next_batch_train(params.batch);
	--Forward/backward
	local word_embedding=caption_embedding_net.deploy:forward({fv_cap,fv_t});
	local tv_cap=caption_encoder_net.deploy:forward({torch.repeatTensor(dummy_state:fill(0),params.batch,1),torch.repeatTensor(dummy_output:fill(0),params.batch,1),word_embedding});
	local scores=multimodal_net.deploy:forward({tv_cap,fv_im});
	local f=criterion_1:forward(scores,labels)+criterion_2:forward(scores:t(),labels);
	local dscores=criterion_1:backward(scores,labels)+criterion_2:backward(scores:t(),labels):t();
	local tmp=multimodal_net.deploy:backward({tv_cap,fv_im},dscores);
	local tmp2=caption_encoder_net.deploy:backward({torch.repeatTensor(dummy_state:fill(0),params.batch,1),torch.repeatTensor(dummy_output:fill(0),params.batch,1),word_embedding},tmp[1]);
	caption_embedding_net.deploy:backward({fv_cap,fv_t},tmp2[3]);
	--summarize f and gradient
	caption_encoder_net.dw:clamp(-2,2);
	running_avg=running_avg*0.95+f*0.05;
end

--Optimization loop
Log('Begin optimizing');
local timer = torch.Timer();
for i=1,max_iter do
	--Save every 10 iterations
	if i%(niter_per_epoch*10)==0 then
		torch.save(paths.concat(basedir,'model',string.format('model_epoch%d.t7',i/(niter_per_epoch))),{caption_encoder_net=caption_encoder_net.net,caption_embedding_net=caption_embedding_net.net,multimodal_net=multimodal_net.net});
	end
	--Print statistics every 1 iteration
	if i%niter_per_epoch==0 then
		collectgarbage();
		Log(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,params.epochs,running_avg,opt_encoder.learningRate,timer:time().real));
	end
	ForwardBackward();
	--Update parameters
	rmsprop(caption_encoder_net.w,caption_encoder_net.dw,opt_encoder);
	rmsprop(caption_embedding_net.w,caption_embedding_net.dw,opt_embedding);
	rmsprop(multimodal_net.w,multimodal_net.dw,opt_multimodal);
	--Learning rate decay
	opt_encoder.learningRate=opt_encoder.learningRate*opt_encoder.decay;
	opt_embedding.learningRate=opt_embedding.learningRate*opt_embedding.decay;
	opt_multimodal.learningRate=opt_multimodal.learningRate*opt_multimodal.decay;
end
