metric={};
function metric.precision(pred,gt)
	local npos=torch.sum(gt);
	local nitems=pred:size(1);
	local shuffle_ind=torch.randperm(nitems, 'torch.LongTensor');
	pred=pred:index(1,shuffle_ind);
	gt=gt:index(1,shuffle_ind);
	local nu,ind1=torch.sort(pred,true)
	local gt_sorted=gt:index(1,ind1);
	local tp=torch.cumsum(gt_sorted);
	local drecall=torch.zeros(nitems):copy(tp)/npos;
	drecall[{{2,nitems}}]=drecall[{{2,nitems}}]-drecall[{{1,nitems-1}}];
	local total=torch.cumsum(torch.ones(nitems));
	return torch.sum(torch.cmul(torch.cdiv(tp,total),drecall)),npos/nitems;
end

function metric.accuracy(pred,gt)
	return torch.sum(pred:eq(gt))/pred:size(1);
end
function metric.accuracy_N(score,gt,n)
	local nitems=gt:size(1);
	local count=0;
	for i=1,nitems do
		local labels={};
		local nu,ind=torch.sort(score[i],true)
		for j=1,n do
			labels[ind[j]]=1;
		end
		if labels[gt[i]]==1 then
			count=count+1;
		end
	end
	return count/nitems;
end

--top N predictions has at least one of the gts.
function metric.accuracy_NM(score,gt,n)
	local nitems=gt:size(1);
	local count=0;
	for i=1,nitems do
		local labels=torch.zeros(score:size(2));
		local nu,ind=torch.sort(score[i],true)
		for j=1,n do
			labels[ind[j]]=1;
		end
		if torch.sum(labels:index(1,gt[i]))>=1 then
			count=count+1;
		end
	end
	return count/nitems;
end

function metric.per_class_accuracy(pred,gt)
	local labels={};
	local nitems=gt:size(1);
	for i=1,nitems do
		if labels[gt[i]]==nil then
			local ind=gt:eq(gt[i]);
			labels[gt[i]]=torch.sum(pred[ind]:eq(gt[ind]))/torch.sum(ind);
		end
	end
	local cnt=0;
	local val=0;
	for k,v in pairs(labels) do
		cnt=cnt+1;
		val=val+v;
	end
	return {val/cnt,labels}
end
return metric;