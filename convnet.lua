require "nn"
require "optim"
matio = require "matio"


net = nn.Sequential()
net:add(nn.Reshape(3, 32, 32))
net:add(nn.SpatialConvolution(3, 6, 4, 4, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6, 9, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(9, 12, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(12, 15, 2, 2, 1, 1, 0, 0))
net:add(nn.ReLU())

net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.Reshape(15*15*15))
net:add(nn.Linear(15*15*15, 1024))
net:add(nn.ReLU())

net:add(nn.Dropout())
net:add(nn.Linear(1024, 512))
net:add(nn.ReLU())

net:add(nn.Dropout())
net:add(nn.Linear(512, 128))
net:add(nn.ReLU())

net:add(nn.Dropout())
net:add(nn.Linear(128, 10))

criterion = nn.CrossEntropyCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.0001
trainer.maxIteration = 30

print (net)



inputs =  {}
outputs = {}
for i = 1, 2 do
        local batch = matio.load(string.format("../data/cifar-10-batches-mat/data_batch_%i.mat", i))
        table.insert(inputs, batch.data)
        table.insert(outputs, batch.labels)
end

traindata = {}
traindata.data = torch.cat(inputs, 1)
traindata.labels = torch.cat(outputs, 1)

shuffled = torch.randperm(traindata.data:size(1)):long()

traindata.data = traindata.data:index(1, shuffled):double()
--traindata.data = traindata.data[{{1, 1000}}]
traindata.labels = traindata.labels:index(1, shuffled):byte() + 1
--traindata.labels = traindata.labels[{{1, 1000}}]


setmetatable(traindata, 
{ __index = function (t, i)
                return { t.data[i], t.labels[i][1] }
            end });


function traindata:size() return self.data:size(1) end



net:training()
trainer:train(traindata)
torch.save("convnet.th", net)
-- net = torch.load("convnet.th")

-- find accuracy on train set
net:evaluate()
classes = {"airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"}
confusion = optim.ConfusionMatrix(classes)
for i = 1, 2000 do
        prediction = net:forward(traindata[i][1])
        scores, indices = torch.sort(prediction, 1, true)
        --print (prediction)
        --print (indices[1], traindata[i][2])
        confusion:add(indices[1], traindata[i][2])
        
end



print (confusion)


net = torch.load("convnet.th")
testdata = matio.load("../data/cifar-10-batches-mat/test_batch.mat")

shuffled = torch.randperm(testdata.data:size(1)):long()

testdata.data = testdata.data:index(1, shuffled):double()
testdata.labels = testdata.labels:index(1, shuffled):long() + 1

setmetatable(testdata, 
{ __index = function (t, i)
                return { t.data[i], t.labels[i][1] }
            end });


function testdata:size() return self.data:size(1) end

classes = {"airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"}
confusion = optim.ConfusionMatrix(classes)

net:evaluate()

for i = 1, testdata:size() do
        prediction = net:forward(testdata[i][1])
        scores, indices = torch.sort(prediction, 1, true)
        confusion:add(indices[1], testdata[i][2])
end

print (confusion)
