import Random
Random.seed!(0)
using LinearAlgebra

sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
linear(x :: Real) = x
σ = sigmoid

η = 0.5
epochs = 10000

inputs  = 2
neurons = 5
outputs = 1

bₕ = hidden_bias = zeros(1, neurons)
bₒ = output_bias = zeros(1, outputs)

wₕ = hidden_layer = rand(inputs,  neurons)
wₒ = output_layer = rand(neurons, outputs)

hidden_activation = sigmoid
output_activation = sigmoid

hidden_activation_derivative = (x) -> sigmoid(x)*(1 - sigmoid(x))
output_activation_derivative = (x) -> sigmoid(x)*(1 - sigmoid(x))

inputs =  [0 0;
           0 1;
           1 0;
           1 1]
targets = [0; 1; 1;  0];

for i=1:epochs
  global hidden_layer, input_layer, output_layer
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], 1, :)
  y = reshape(targets[j,:], 1, :)
  
  ### Feed forward ###
  x̄ = sum(x * wₕ  .+ bₕ; dims=1)
  x̂ = hidden_activation.(x̄)
                                               
  ȳ = sum(x̂ * wₒ .+ bₒ; dims=1)
  ŷ = output_activation.(ȳ)

  E = loss = sum(0.5*(ŷ - y).^2)
  @show loss

  ### Back propagation ###
  ∂E_∂ŷ  = -(ŷ - y)
  ∂ŷ_∂wₒ = output_activation_derivative.(ŷ) .* ŷ
  ∂E_∂wₒ = δ = ∂E_∂ŷ .* ∂ŷ_∂wₒ
  wₒ    -= η * δ

  ∂E_∂x̂  = ...
  ∂x̂_∂wₕ = ...
  ∂E_∂wₕ = δ = ∂E_∂x̂ .* ∂x̂_∂wₕ
  wₕ.   -= η * δ
end
