import Random
Random.seed!(0)
using LinearAlgebra

∑ = sum
sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
linear(x :: Real) = x
σ = sigmoid

η = 0.95
epochs = 10000

inputs  = 2
neurons = 5
outputs = 1

bₕ = hidden_bias = zeros(1, neurons)
bₒ = output_bias = zeros(1, outputs)

wₕ = hidden_layer = rand(inputs,  neurons)
wₒ = output_layer = rand(neurons, outputs)

inputs =  [0 0;
           0 1;
           1 0;
           1 1]
targets = [0; 1; 1;  0];

for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], 1, :)
  y = reshape(targets[j,:], 1, :)
  
  ### Feed forward ###
  x̄ = ∑(x * wₕ .+ bₕ; dims=1)
  x̂ = σ.(x̄)
                                               
  ȳ = ∑(x̂ * wₒ .+ bₒ; dims=1)
  ŷ = σ.(ȳ)

  ε = 0.5*(y - ŷ).^2
  E = loss = ∑(ε)
  @show loss

  ### Back propagation ###
  #@show size(x)
  #@show size(x̂)
  #@show size(ŷ)
  ∂ȳ_∂bₒ = 1.
  ∂ȳ_∂wₒ = reshape(x̂, :, 1)
  ∂ŷ_∂ȳ  = σ.(ŷ) .* (1 .- σ.(ŷ))
  ∂E_∂ŷ  = -(y - ŷ)
  #@show size(∂ȳ_∂wₒ)
  #@show size(∂ŷ_∂ȳ)
  #@show size(∂E_∂ŷ)
  #@show size(wₒ)
  #@show size(bₒ)
  ∂E_∂wₒ = ∂ȳ_∂wₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  ∂E_∂bₒ = ∂ȳ_∂bₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  #@show size(∂E_∂wₒ)
  #@show size(∂E_∂bₒ)
  wₒ    -= η * ∂E_∂wₒ
  bₒ    -=2η * ∂E_∂bₒ
  #println("========")
  ∂ȳ_∂x̂  = wₒ
  ∂ŷ_∂ȳ  = σ.(ŷ) .* (1 .- σ.(ŷ))
  ∂E_∂x̂  = ∑(∂ŷ_∂ȳ .* ∂ȳ_∂x̂)
  ∂x̂_∂x̄  = σ.(x̂) .* (1 .- σ.(x̂))
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
