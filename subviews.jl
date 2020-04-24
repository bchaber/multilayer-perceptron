function subviews(u, ns...)
   vs, s = [], 0
   for n in ns
     N = prod(n)
     v = reshape(view(u, (1:N) .+ s), n...)
     s += N
     push!(vs, v)
   end    
   vs   
end 
