import Random: shuffle, shuffle!, seed!
seed!(0)

include("dual.jl")
include("subviews.jl")
mean_squared_loss(y, ŷ) = sum(0.5(y - ŷ).^2)
dense(w, b, n::Integer, m::Integer, v, activation::Function) = 
  activation(reshape(w, n, m) * v .+ reshape(b, n, 1))
σ(x) = @. one(x) / (one(x) + exp(-x))
ReLU(x) = @. max(zero(x), x)
swish(x) = @. x / (one(x) + exp(-x))
linear(x) = x
softmax(x) = exp.(x) ./ sum(exp.(x))

input_neurons  = 2
hidden_neurons = 10
output_neurons = 1

parameters = zeros(hidden_neurons*input_neurons  + hidden_neurons +
                   output_neurons*hidden_neurons + output_neurons)
wh, _, wo, _ = subviews(parameters,
  (hidden_neurons*input_neurons), (hidden_neurons),
  (output_neurons*hidden_neurons),(output_neurons))
wh .= randn(hidden_neurons*input_neurons)
wo .= randn(output_neurons*hidden_neurons)

η = 0.1
epochs = 5000
batch_size = 1

function net(x, wh, bh, wo, bo)
    x̂ = dense(wh, bh, hidden_neurons,  input_neurons, x, swish)
    ŷ = dense(wo, bo, output_neurons, hidden_neurons, x̂, linear)
end

function loss(x, y, wh, bh, wo, bo)
	ŷ = net(x, wh, bh, wo, bo)
	mean_squared_loss(y, ŷ)
end

include("datasets/hagan_smart_sensor.jl")
test_size  = 10
train_size = 57
data_size  = train_size + test_size
train_set  = shuffle(1:data_size)[1:train_size]
test_set   = setdiff(1:data_size, train_set)

dloss_wh(x, y, wh, bh, wo, bo) = vec(J(w -> loss(x, y, w,  bh, wo, bo), wh));
dloss_bh(x, y, wh, bh, wo, bo) = vec(J(b -> loss(x, y, wh, b,  wo, bo), bh));
dloss_wo(x, y, wh, bh, wo, bo) = vec(J(w -> loss(x, y, wh, bh, w,  bo), wo));
dloss_bo(x, y, wh, bh, wo, bo) = vec(J(b -> loss(x, y, wh, bh, wo, b),  bo));

function test(parameters, test_set)
  wh, bh, wo, bo = subviews(parameters,
    (hidden_neurons*input_neurons), (hidden_neurons),
    (output_neurons*hidden_neurons),(output_neurons))
  Et  = zero(0.)
  for j = test_set
    x   = reshape( inputs[j,:], :, 1)
    y   = reshape(targets[j,:], :, 1)

    Et += loss(x, y, wh, bh, wo, bo)
  end
  return Et/length(test_set)
end

function train(parameters, train_set)
  wh, bh, wo, bo = subviews(parameters,
    (hidden_neurons*input_neurons), (hidden_neurons),
    (output_neurons*hidden_neurons),(output_neurons))
  ∇E = zeros(length(parameters))
  for j = train_set
    x   = reshape( inputs[j,:], :, 1)
    y   = reshape(targets[j,:], :, 1)
    ŷ   = net(x, wh, bh, wo, bo)

    Ewh = dloss_wh(x, y, wh, bh, wo, bo)
    Ebh = dloss_bh(x, y, wh, bh, wo, bo)
    Ewo = dloss_wo(x, y, wh, bh, wo, bo)
    Ebo = dloss_bo(x, y, wh, bh, wo, bo)
    ∇E.+= vcat(Ewh, Ebh, Ewo, Ebo)
  end
  return ∇E/length(train_set)
end

for i=1:epochs
  shuffle!(train_set)
  ∇E = train(parameters, train_set[1:batch_size])
  parameters .-= η*∇E
  println(i, "\t", test(parameters, test_set))
end
println("FINAL", "\t", test(parameters, 1:data_size))
