--Simple MemNN that runs like LSTM<-Mem. 
--CPU: yes
--Read: yes
--Write: no
--Mem is batch x address x data

require 'nn'
require 'nngraph'
MemNN={};

function MemNN:MemNN(rnn_unit,n,nhaddress,nhdata)
	local init_output=nn.Identity()();
	local init_state=nn.Identity()();
	local data=nn.Identity()();
	local state=init_state;
	local output=init_output;
	local input=nn.View(-1,nhdata)(nn.MM()({nn.View(-1,1,nhaddress)(output),data}));
	for i=1,n-1 do
		local lstm_node=rnn_unit:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var')({state,input});
		state=nn.SelectTable(1)(lstm_node);
		output=nn.SelectTable(2)(lstm_node);
		input=nn.View(-1,nhdata)(nn.MM()({nn.View(-1,1,nhaddress)(output),data}));
	end
	local lstm_node=rnn_unit:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var')({state,input});
	state=nn.SelectTable(1)(lstm_node);
	return nn.gModule({init_state,init_output,data},{state});
end


function MemNN:new(unit,n,nhaddress,nhdata,gpu)
	--unit: {state,input}->{state,output}
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = MemNN
	setmetatable(net,netmeta);
	--stuff
	if gpu then
		net.net=unit:cuda();
	else
		net.net=unit;
	end
	net.w,net.dw=net.net:getParameters();
	net.n=n;
	net.nhaddress=nhaddress;
	net.nhdata=nhdata;
	if gpu then
		net.deploy=self:MemNN(unit,n,nhaddress,nhdata):cuda();
	else
		net.deploy=self:MemNN(unit,n,nhaddress,nhdata)
	end
	collectgarbage();
	net.cell_inputs={};
	return net;
end
function MemNN:training()
	self.net:training();
	self.deploy:training();
end
function MemNN:evaluate()
	self.net:evaluate();
	self.deploy:evaluate();
end
function MemNN:clearState()
	self.net:clearState();
	self.deploy:clearState();
end
function MemNN:forward(a)
	return self.deploy:forward(a);
end
function MemNN:backward(a,b)
	local c=self.deploy:backward(a,b);
	return c;
end

MemNN.unit={};
--nhinput: input size
--nh: hidden size
--noutput: output size
--n: number of stacks (layers)
--dropout: output dropout probability
function MemNN.unit.lstm(nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local h_prev=nn.Identity()(); -- batch x (2nh), combination of past cell state and hidden state
	local input=nn.Identity()(); -- batch x nhword, input embeddings
	local prev_c={};
	local prev_h={};
	for i=1,n do
		prev_c[i]=nn.Narrow(2,2*(i-1)*nh+1,nh)(h_prev);
		prev_h[i]=nn.Narrow(2,(2*i-1)*nh+1,nh)(h_prev);
	end
	local c={};
	local h={};
	local mixed={};
	for i=1,n do
		local x;
		local input_size;
		if i==1 then
			x=input;
			input_size=nhinput;
		else
			x=h[i-1];
			input_size=nh;
		end
		local controls=nn.Linear(input_size+nh,4*nh)(nn.JoinTable(1,1)({x,prev_h[i]}));
		--local sigmoid_chunk=nn.Sigmoid()(nn.Narrow(2,1,3*nh)(controls));
		local data_chunk=nn.Tanh()(nn.Narrow(2,3*nh+1,nh)(controls));
		local in_gate=nn.Sigmoid()(nn.Narrow(2,1,nh)(controls));
		local out_gate=nn.Sigmoid()(nn.Narrow(2,nh+1,nh)(controls));
		local forget_gate=nn.Sigmoid()(nn.Narrow(2,2*nh+1,nh)(controls));
		c[i]=nn.CAddTable()({nn.CMulTable()({forget_gate,prev_c[i]}),nn.CMulTable()({in_gate,data_chunk})});
		h[i]=nn.CMulTable()({out_gate,nn.Tanh()(c[i])});
		table.insert(mixed,c[i]);
		table.insert(mixed,h[i]);
	end
	local h_current=nn.JoinTable(1,1)(mixed);
	local output=nn.Linear(nh,noutput)(nn.Dropout(dropout)(h[n]));
	return nn.gModule({h_prev,input},{h_current,output});
end

return MemNN