using MLDatasets

images, classes = FashionMNIST.traindata()
inputs  = images ./ 255 .- 0.5
targets = zeros(10, 60_000)
for i=1:60_000
  targets[classes[i] + 1, i] = 1
end