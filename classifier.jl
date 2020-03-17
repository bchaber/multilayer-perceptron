import Random: shuffle, seed!
seed!(0)

η = 0.001 # learning rate
epochs = 500

batch_size     = 1
input_neurons  = 4
hidden_neurons = 8
output_neurons = 3

wb = zeros(hidden_neurons*input_neurons  + hidden_neurons +
           output_neurons*hidden_neurons + output_neurons)
∂E = zeros(hidden_neurons*input_neurons  + hidden_neurons +
           output_neurons*hidden_neurons + output_neurons)

include("subviews.jl")
wₕ, bₕ, wₒ, bₒ = subviews(wb,
  (hidden_neurons, input_neurons),  (hidden_neurons,1),
  (output_neurons, hidden_neurons), (output_neurons,1));
∂wₕ, ∂bₕ, ∂wₒ, ∂bₒ = subviews(∂E,
  (hidden_neurons, input_neurons),  (hidden_neurons,1),
  (output_neurons, hidden_neurons), (output_neurons,1));

include("nn.jl")
hidden = FullyConnectedLayer{_ReLU}(wₕ, bₕ, ∂wₕ, ∂bₕ)
output = FullyConnectedLayer{_softmax}(wₒ, bₒ, ∂wₒ, ∂bₒ)
nn = NeuralNetwork{cross_entropy_loss}(hidden, output)
wₕ .= randn(hidden_neurons, input_neurons)
wₒ .= randn(output_neurons, hidden_neurons)

include("optimizers.jl")
optimizer = GradientDescent(η)

include("datasets/iris.jl")
test_size  =  10
train_size = 140
data_size  = train_size + test_size
train_set  = shuffle(1:data_size)[1:train_size]
test_set   = setdiff(1:data_size, train_set)

function test(wb′)
  wb .= wb′
  loss(nn, inputs, targets, test_set)
end

function train(wb′)
  wb .= wb′
  ∇E = zeros(size(∂E))
  for _=1:batch_size
    j = rand(train_set)
    x = reshape( inputs[j,:], :, 1)
    y = reshape(targets[j,:], :, 1)

    feedforward!(nn, x)
    backpropagate!(nn, y)
    ∇E += ∂E
  end
  return ∇E/batch_size
end

function optimize!(wb)
  wb .= step!(optimizer, test, train, copy(wb))
end

for i=1:epochs
  optimize!(wb)
  println(i, "\t", test(wb))
end

test_set = 1:data_size
println("FINAL", "\t", test(wb))
