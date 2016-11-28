require "nn"
require "optim"


net = nn.Sequential()
net:add(nn.Linear(3*32*32, 1024))
net:add(nn.ReLU())
net:add(nn.Linear(1024, 512))
net:add(nn.ReLU())
net:add(nn.Linear(512, 256))
net:add(nn.ReLU())
net:add(nn.Linear(256, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128, 10))

criterion = nn.CrossEntropyCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.0001
trainer.maxIteration = 5

print (net)
matio = require "matio"
traindata = matio.load("../data/cifar-10-batches-mat/data_batch_1.mat")

shuffled = torch.randperm(traindata.data:size(1)):long()

traindata.data = traindata.data:index(1, shuffled):double()
traindata.data = traindata.data[{{1, 2000}}]
traindata.labels = traindata.labels:index(1, shuffled):long() + 1
traindata.labels = traindata.labels[{{1, 2000}}]

setmetatable(traindata, 
{ __index = function (t, i)
                return { t.data[i], t.labels[i][1] }
            end });


function traindata:size() return self.data:size(1) end




trainer:train(traindata)
torch.save("simple.th", net)
-- net = torch.load("simple.th")

-- find accuracy on train set
classes = {"airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"}
confusion = optim.ConfusionMatrix(classes)
for i = 1, 1000 do
        prediction = net:forward(traindata[i][1])
        scores, indices = torch.sort(prediction, 1, true)
        --print (prediction)
        --print (indices[1], traindata[i][2])
        confusion:add(indices[1], traindata[i][2])
end



print (confusion)





