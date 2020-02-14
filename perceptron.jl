import Random
Random.seed!(0)

sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
linear(x :: Real)  = x
ReLU(x :: Real) = max(0, x)
σ = sigmoid
∑ = sum

η = 0.95 # learning rate
epochs = 10000

input_neurons  = 4
hidden_neurons = 5
output_neurons = 3

bₕ = hidden_bias = zeros(1, hidden_neurons)
bₒ = output_bias = zeros(1, output_neurons)
wₕ = hidden_layer = rand(input_neurons,  hidden_neurons) .- 0.5
wₒ = output_layer = rand(hidden_neurons, output_neurons) .- 0.5
include("iris.jl")
for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], 1, :)
  y = reshape(targets[j,:], 1, :)

  ### Feed forward ###
  x̄ = x * wₕ .+ bₕ
  x̂ = σ.(x̄)
  
  ȳ = x̂ * wₒ .+ bₒ
  ŷ = σ.(ȳ)
  
  E = ∑(0.5*(y - ŷ).^2)

  ### Back propagation ###
  ∂ȳ_∂bₒ = 1.
  ∂ȳ_∂wₒ = reshape(x̂, :, 1)
  ∂ŷ_∂ȳ  = ŷ .* (1 .- ŷ)
  ∂E_∂ŷ  = -(y - ŷ)
  ∂E_∂wₒ = ∂ȳ_∂wₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ
  ∂E_∂bₒ = ∂ȳ_∂bₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ

  ∂x̄_∂bₕ = 1.
  ∂x̄_∂wₕ = reshape(x, :, 1)
  ∂x̂_∂x̄  = x̂ .* (1 .- x̂)
  ∂ȳ_∂x̂  = wₒ
  ∂E_∂x̂  = ∑(∂ȳ_∂x̂ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ; dims=2)'
  ∂E_∂x̄  = ∂x̂_∂x̄ .* ∂E_∂x̂
  ∂E_∂wₕ = ∂x̄_∂wₕ * ∂E_∂x̄
  ∂E_∂bₕ = ∂x̄_∂bₕ * ∂E_∂x̄
  wₒ    -= η * ∂E_∂wₒ
  bₒ    -= η * ∂E_∂bₒ
  wₕ    -= η * ∂E_∂wₕ
  bₕ    -= η * ∂E_∂bₕ
  println(i, "\t", E)
end
