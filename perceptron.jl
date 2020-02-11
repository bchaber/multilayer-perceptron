import Random
Random.seed!(0)
using LinearAlgebra

∑ = sum
sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
linear(x :: Real) = x
σ = sigmoid

η = 0.95
epochs = 10000

inputs  = 3
neurons = 2
outputs = 2

bₕ = hidden_bias = zeros(1, neurons)
bₒ = output_bias = zeros(1, outputs)
bₕ = [0.5 0.5]
bₒ = [0.5 0.5]
wₕ = hidden_layer = rand(inputs,  neurons)
wₒ = output_layer = rand(neurons, outputs)
wₕ = [0.1 0.2;
      0.3 0.4;
      0.5 0.6]
wₒ = [0.7 0.8;
      0.9 0.1]
inputs =  [1 4 5]
targets = [0.1 0.05]

for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], 1, :)
  y = reshape(targets[j,:], 1, :)
  #@show x
  ### Feed forward ###
  x̄ = ∑(x * wₕ; dims=1) .+ bₕ
  x̂ = σ.(x̄)
  #@show x̄
  #@show x̂

  ȳ = ∑(x̂ * wₒ; dims=1) .+ bₒ
  ŷ = σ.(ȳ)
  #@show ȳ
  #@show ŷ

  ε = 0.5*(y - ŷ).^2
  E = loss = ∑(ε)
  @show loss

  ### Back propagation ###
  #@show size(x)
  #@show size(x̂)
  #@show size(ŷ)
  ∂ȳ_∂bₒ = 1.
  ∂ȳ_∂wₒ = reshape(x̂, :, 1)
  ∂ŷ_∂ȳ  = ŷ .* (1 .- ŷ)
  ∂E_∂ŷ  = -(y - ŷ)
  #@show ∂ȳ_∂wₒ
  #@show ∂ŷ_∂ȳ
  #@show ∂E_∂ŷ
  #@show size(wₒ)
  #@show size(bₒ)
  ∂E_∂wₒ = ∂ȳ_∂wₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  ∂E_∂bₒ = ∂ȳ_∂bₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  #@show ∂E_∂wₒ
  #@show size(∂E_∂wₒ)
  #@show size(∂E_∂bₒ)
  wₒ    -= η * ∂E_∂wₒ
  bₒ    -=2η * ∂E_∂bₒ
  #println("========")
  ∂ȳ_∂x̂  = wₒ
  ∂ŷ_∂ȳ  = ŷ .* (1 .- ŷ)
  ∂E_∂x̂  = ∑(∂ŷ_∂ȳ .* ∂ȳ_∂x̂)
  ∂x̂_∂x̄  = x̂ .* (1 .- x̂)
  ∂x̄_∂bₕ = 1.
  ∂x̄_∂wₕ = reshape(x, :, 1)
  #@show size(∂ȳ_∂x̂)
  #@show size(∂ŷ_∂ȳ)
  #@show size(∂x̂_∂x̄)
  #@show size(∂x̄_∂wₕ)
  #@show size(wₕ)
  #@show size(bₕ)
  ∂E_∂wₕ = ∂x̄_∂wₕ * ∂x̂_∂x̄ .* ∂E_∂x̂
  ∂E_∂bₕ = ∂x̄_∂bₕ * ∂x̂_∂x̄ .* ∂E_∂x̂
  #@show size(∂E_∂wₕ)
  #@show size(∂E_∂bₕ)
  wₕ    -= η * ∂E_∂wₕ
  bₕ    -=2η * ∂E_∂bₕ
end
