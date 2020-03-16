Float = Float64
abstract type Layer end
abstract type ActivationFunction end
abstract type _sigmoid <: ActivationFunction end
abstract type _softmax <: ActivationFunction end
abstract type _linear  <: ActivationFunction end
abstract type _ReLU    <: ActivationFunction end
abstract type _tanh    <: ActivationFunction end

sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
softmax(x :: Array{<:Real,2}) = exp.(x) ./ sum(exp.(x))
linear(x :: Real)  = x
ReLU(x :: Real) = max(0, x)
tanh(x :: Real) = 2 / (1 + exp(-2x)) - 1

struct FullyConnectedLayer{T <: ActivationFunction} <: Layer
  w # weights
  b # bias
  u # input
  ū # sum
  û # output
  # derivatives
  ūb
  ūw
  ūu
  ûū
  Ew
  Eb
end
function FullyConnectedLayer{T}(w, b, Ew, Eb) where T
  n, m = size(w)

  FullyConnectedLayer{T}(w, b,
	zeros(m,1), zeros(n,1), zeros(n,1),
	zeros(n,1), zeros(1,m), zeros(m,n), zeros(n,1),
	Ew, Eb)
end

struct NeuralNetwork
  hidden :: Layer
  output :: Layer
end

function mean_squared_error(nn::NeuralNetwork, inputs, targets, ks)
  Eₜ  = 0.
  for k = ks
    x   = reshape( inputs[k,:], :, 1)
    y   = reshape(targets[k,:], :, 1)
    ŷ   = feedforward!(nn, x)
    Eₜ += sum(0.5(y - ŷ).^2)
  end
  return Eₜ/length(ks)
end

function feedforward!(nn::NeuralNetwork, x)
  x̂ = feedforward!(nn.hidden, x)
  ŷ = feedforward!(nn.output, x̂)
end

function feedforward!(fc::FullyConnectedLayer{_tanh}, u)
  fc.u .= u
  fc.ū .= fc.w * u + fc.b
  fc.û .= tanh.(fc.ū)
end

function feedforward!(fc::FullyConnectedLayer{_linear}, u)
  fc.u .= u
  fc.ū .= fc.w * u + fc.b
  fc.û .= linear.(fc.ū)
end

function backpropagate!(fc::FullyConnectedLayer{_tanh}, u)
  fc.ūb .= ê = ones(size(fc.ūb))
  fc.ūw .= fc.u |> transpose
  fc.ūu .= fc.w |> transpose
  fc.ûū .= 1.0 .- fc.û .^ 2
end

function backpropagate!(fc::FullyConnectedLayer{_linear}, u)
  fc.ūb .= ê = ones(size(fc.ūb))
  fc.ūw .= fc.u |> transpose
  fc.ūu .= fc.w |> transpose
  fc.ûū .= ê = ones(size(fc.ûū))
end

function backpropagate!(nn::NeuralNetwork, y)
  backpropagate!(nn.output, y)
  backpropagate!(nn.hidden, y)
  ŷ = nn.output.û
  E = sum(0.5(y - ŷ).^2)
  
  ŷȳ = nn.output.ûū
  ȳx̂ = nn.output.ūu
  x̂x̄ = nn.hidden.ûū

  Eŷ  =-(y - ŷ)
  Eȳ  = ŷȳ * Eŷ
  Ex̂  = ȳx̂ * Eȳ
  Ex̄  = Ex̂.* x̂x̄
  
  ȳwₒ = nn.output.ūw
  ȳbₒ = nn.output.ūb
  x̄wₕ = nn.hidden.ūw
  x̄bₕ = nn.hidden.ūb
  nn.output.Ew .= Eȳ * ȳwₒ
  nn.output.Eb .= Eȳ.* ȳbₒ
  nn.hidden.Ew .= Ex̄ * x̄wₕ
  nn.hidden.Eb .= Ex̄.* x̄bₕ
end
