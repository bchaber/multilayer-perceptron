import LinearAlgebra: diagm
Float = Float64
abstract type Layer end
abstract type ActivationFunction end
abstract type swish   <: ActivationFunction end
abstract type sigmoid <: ActivationFunction end
abstract type softmax <: ActivationFunction end
abstract type linear  <: ActivationFunction end
abstract type ReLU    <: ActivationFunction end
abstract type tanh    <: ActivationFunction end
abstract type LossFunction end
abstract type mean_squared_loss <: LossFunction end
abstract type cross_entropy_loss <: LossFunction end

include("dense.jl")

struct NeuralNetwork{T <: LossFunction}
  hidden :: Layer
  output :: Layer
end

loss(nn::NeuralNetwork{mean_squared_loss}, y, ŷ)  = sum(0.5(y - ŷ).^2)
loss(nn::NeuralNetwork{cross_entropy_loss}, y, ŷ) = sum(-y .* log.(ŷ))

function loss(nn::NeuralNetwork{T}, inputs, targets, ks) where T
  Eₜ  = 0.
  for k = ks
    x   = reshape( inputs[k,:], :, 1)
    y   = reshape(targets[k,:], :, 1)
    ŷ   = feedforward!(nn, x)
    Eₜ += loss(nn, y, ŷ)
  end
  return Eₜ/length(ks)
end

∂loss(nn::NeuralNetwork{mean_squared_loss}, y, ŷ)  = -(y - ŷ)
∂loss(nn::NeuralNetwork{cross_entropy_loss}, y, ŷ) = -y ./ ŷ

function feedforward!(nn::NeuralNetwork, x)
  x̂ = feedforward!(nn.hidden, x)
  ŷ = feedforward!(nn.output, x̂)
end

function backpropagate!(nn::NeuralNetwork{T}, y) where T
  backpropagate!(nn.output)
  backpropagate!(nn.hidden)
  ŷ = nn.output.û
  
  ŷȳ = nn.output.ûū
  ȳx̂ = nn.output.ūu
  x̂x̄ = nn.hidden.ûū

  Eŷ  = ∂loss(nn, y, ŷ)
  Eȳ  = ŷȳ * Eŷ
  Ex̂  = ȳx̂ * Eȳ
  Ex̄  = x̂x̄ * Ex̂
  
  ȳwₒ = nn.output.ūw
  ȳbₒ = nn.output.ūb
  x̄wₕ = nn.hidden.ūw
  x̄bₕ = nn.hidden.ūb
  nn.output.Ew .= Eȳ * ȳwₒ
  nn.output.Eb .= Eȳ.* ȳbₒ
  nn.hidden.Ew .= Ex̄ * x̄wₕ
  nn.hidden.Eb .= Ex̄.* x̄bₕ
end
