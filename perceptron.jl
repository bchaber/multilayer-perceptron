import Random
Random.seed!(0)
using LinearAlgebra

function pprint(A)
  io = IOContext(stdout)
  io = IOContext(io, :limit=>true)
  show(io, "text/plain", A)
  println()
end

∑ = sum
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
inputs =  [0 0;
           0 1;
           1 0;
           1 1]
targets = [0; 1; 1; 0]
for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
#print("wₕ = "); pprint(wₕ)
#print("wₒ = "); pprint(wₒ)
  #j = (i-1) % 4 + 1
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], 1, :)
  y = reshape(targets[j,:], 1, :)
  ### Feed forward ###
  x̄ = ∑(x * wₕ; dims=1) .+ bₕ
  x̂ = σ.(x̄)
  
  ȳ = ∑(x̂ * wₒ; dims=1) .+ bₒ
  ŷ = σ.(ȳ)
  
  ε = 0.5*(y - ŷ).^2
  E = ∑(ε)

  ### Back propagation ###
  ∂ȳ_∂bₒ = 1.
  ∂ȳ_∂wₒ = reshape(x̂, :, 1)
  ∂ŷ_∂ȳ  = ŷ .* (1 .- ŷ)
  ∂E_∂ŷ  = -(y - ŷ)
  ∂E_∂wₒ = ∂ȳ_∂wₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  ∂E_∂bₒ = ∂ȳ_∂bₒ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ 
  #println("========")
  ∂ȳ_∂x̂  = wₒ
  ∂x̂_∂x̄  = x̂ .* (1 .- x̂)
  ∂x̄_∂bₕ = 1.
  ∂x̄_∂wₕ = reshape(x, :, 1)
  ∂E_∂x̂ = ∂ȳ_∂x̂ .* ∂ŷ_∂ȳ .* ∂E_∂ŷ
  ∂E_∂x̄ = ∂x̂_∂x̄ .* ∂E_∂x̂'
  ∂E_∂wₕ = ∂x̄_∂wₕ * ∂E_∂x̄
  ∂E_∂bₕ = ∂x̄_∂bₕ * ∂E_∂x̄

  wₒ    -= η * ∂E_∂wₒ
  bₒ    -= η * ∂E_∂bₒ
  wₕ    -= η * ∂E_∂wₕ
  bₕ    -= η * ∂E_∂bₕ
  q₁ = norm(η * ∂E_∂wₒ)
  q₂ = norm(η * ∂E_∂wₕ)
  println("$i\t$E\t$q₁\t$q₂")#@show loss
end
