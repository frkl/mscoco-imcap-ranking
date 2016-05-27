cmd = torch.CmdLine();
cmd:text('Compute scores from a image-caption ranking model');
cmd:text('Options')
cmd:option('-session','','Session name');
cmd:option('-data','../dataset/mscoco-imcap/dataset_test.t7','Dataset for evaluation');
cmd:option('-K',1000,'Report results on ranking first K images. Typically K=1000. Some papers report K=5000');
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM)');
params=cmd:parse(arg);
--print(params)

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'metric'
MemNN=require('word_MemNN');

print('Initializing session')
Session=require('session_manager');
session=Session:init('./sessions');
params_train=session:get_params(params.session);
basedir=paths.concat('./sessions',params.session);

log_file=paths.concat(basedir,string.format('log_test_%d.txt',params.K));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end

Log('Loading dataset');
dataset=torch.load(params.data);
ncaptions_per_im=dataset['ncaptions_per_im'];
vocabulary_size=#dataset['caption_dictionary'];
nhsent=params_train.nlayers*params_train.nh*2;
dummy_state=torch.DoubleTensor(nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params_train.nhtime):fill(0):cuda();

Log('Computing scores');
ncaps=math.min(dataset['caption_tokens']:size(1),params.K*ncaptions_per_im);
nim=math.min(dataset['image_fvs']:size(1),params.K);
--Compute scores for every 10 iterations
for iter=10,params_train.epochs,10 do
	collectgarbage();
	Log(string.format('Iter %d',iter));
	print('Loading models');
	function wrap_net(net,gpu)
		local d={};
		if gpu then
			d.net=net:cuda();
		else
			d.net=net;
		end
		d.w,d.dw=d.net:getParameters();
		d.deploy=d.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
		return d;
	end
	local model=torch.load(paths.concat(basedir,'model',string.format('model_epoch%d.t7',iter)));
	local caption_encoder_net=MemNN:new(model.caption_encoder_net,params_train.ntimes,params_train.nhtime,params_train.nhword,true);
	local caption_embedding_net=wrap_net(model.caption_embedding_net,true);
	local multimodal_net=wrap_net(model.multimodal_net,true);
	caption_embedding_net.deploy:evaluate();
	caption_encoder_net.deploy:evaluate();
	multimodal_net.deploy:evaluate();
	--Batch function
	function dataset:next_batch_test(s1,e1,s2,e2)
		local timer = torch.Timer();
		local ims=dataset['image_fvs']:size(1);
		local fv_cap=dataset['caption_tokens'][{{s1,e1},{}}]+1;
		local fv_t=torch.repeatTensor(torch.range(1,dataset['caption_tokens']:size(2)):long(),e1-s1+1,1):reshape((e1-s1+1)*dataset['caption_tokens']:size(2));
		local fv_im=dataset['image_fvs'][{{s2,e2},{}}];
		return fv_cap,fv_t,fv_im:cuda();
	end
	--Compute scores
	running_avg=0;
	function Forward(s1,e1,s2,e2)
		local timer = torch.Timer();
		--grab a batch--
		local fv_cap,fv_t,fv_im=dataset:next_batch_test(s1,e1,s2,e2);
		local word_embedding=caption_embedding_net.deploy:forward({fv_cap,fv_t});
		local tv_cap=caption_encoder_net.deploy:forward({torch.repeatTensor(dummy_state:fill(0),e1-s1+1,1),torch.repeatTensor(dummy_output:fill(0),e1-s1+1,1),word_embedding});
		local scores=multimodal_net.deploy:forward({tv_cap,fv_im});
		return scores:double();
	end
	
	print('Computing scores');
	local scores=torch.DoubleTensor(ncaps,nim);
	for j=1,nim,params.batch do
		for i=1,ncaps,params.batch do
			print(string.format('Image %d/%d, Caption %d/%d',j,nim,i,ncaps));
			ri=math.min(i+params.batch-1,ncaps);
			rj=math.min(j+params.batch-1,nim);
			scores[{{i,ri},{j,rj}}]=Forward(i,ri,j,rj);
		end
	end
	
	print('Evaluating performance');
	local gt_im=torch.zeros(params.K,ncaptions_per_im);
	local gt_text=torch.zeros(params.K,ncaptions_per_im);
	for i=1,params.K do
		gt_im[i]:fill(i);
		gt_text[i]=torch.range(1,ncaptions_per_im)+(i-1)*ncaptions_per_im;
	end
	gt_im=gt_im:view(-1):long();
	gt_text=gt_text:long();
	
	im_r_1=metric.accuracy_N(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}],gt_im,1);
	im_r_5=metric.accuracy_N(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}],gt_im,5);
	im_r_10=metric.accuracy_N(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}],gt_im,10);
	
	cap_r_1=metric.accuracy_NM(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}]:t(),gt_text,1);
	cap_r_5=metric.accuracy_NM(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}]:t(),gt_text,5);
	cap_r_10=metric.accuracy_NM(scores[{{1,params.K*ncaptions_per_im},{1,params.K}}]:t(),gt_text,10);
	
	Log(string.format('Image retrieval: \t%f\t%f\t%f',im_r_1,im_r_5,im_r_10));
	Log(string.format('Caption retrieval: \t%f\t%f\t%f',cap_r_1,cap_r_5,cap_r_10));
	
	collectgarbage();
end
