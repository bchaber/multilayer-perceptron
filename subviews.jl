function subviews(u, ws...)
  vs, s = [], 0
  for w in ws
  	r, c = w
  	n = r*c
    v = view(u, (1:n) .+ s)
    v = reshape(v, r, c)
    s += n
    push!(vs, v)
  end
  vs
end