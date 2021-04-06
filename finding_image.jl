### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 0b6609a2-968c-11eb-3684-ad836c52007b
begin
	using Base.Threads
	using BenchmarkTools
	using LinearAlgebra
	using BenchmarkTools
	using Images
	using ImageView
	using Ditherings
	using PlutoUI
	using DSP #for conv
	using LoopVectorization, OffsetArrays
	using CUDA
	using Plots
end

# ╔═╡ d12c1e0d-6663-4372-ae5a-9a7e709ee518
begin
    n_generations = 6
	batch_size = 50
	MAX_MUTATIONS = 8
end

# ╔═╡ 0916b0a7-fe88-4d23-94c6-0132dedee62f
begin
	monalisa_img = mktemp() do fn,f
       download("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/483px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg", fn)
       load(fn)
end

    monalisa_img_gs = Gray.(monalisa_img)
    monalisa_img_dithered = Ditherings.FloydSteinbergDither4Sample(monalisa_img_gs,       Ditherings.ZeroOne)
    lisa_contrast_img = map(x -> round(x), Gray.(monalisa_img_dithered))
    lisa_img = Gray.(Int8.(lisa_contrast_img))
end

# ╔═╡ 95de85db-1bf3-43e5-978f-97c736853ef7
lisa_img

# ╔═╡ e0a49733-2b1f-4aed-ae7b-5d33b7aa6e99
function tileimg(img::AbstractArray, num::Integer)
	outarray = Array{typeof(img)}(undef, num)
	for i in 1:num
		outarray[i] = copy(img)
	end
	outarray
end

# ╔═╡ 950c80a8-0860-4c9d-9081-218b8e88f3c5
lisa_loaf = tileimg(UInt8.(lisa_img), batch_size)

# ╔═╡ ab895b8d-028e-4091-820f-93bce888b3ba
liveordie(count::UInt8, current::UInt8) = 
(current == 0x00 && count==0x03) || (current == 0x01 && (0x01 < count <=0x03))


# ╔═╡ f8386e21-3d9c-4363-8352-e90858b1a47d
function convGOL_conv(A::AbstractMatrix, kern::AbstractMatrix)
	
	counts = conv(A, kern)[2:end-1, 2:end-1]
	out    = zeros(eltype(A), size(A))
	#size(counts) should == size(A)
	@inbounds for i in CartesianIndices(size(A))
		out[i] = liveordie(UInt8(counts[i]), A[i])
	end
	out
end


# ╔═╡ 1b348db5-6185-4c8a-a6d5-979fa03f9f61
function convGOL_dot(A::AbstractMatrix, kern::AbstractMatrix)
	b = CartesianIndex(1,1)
	out = similar(A)
	@inbounds for i in CartesianIndices(A)[2:end-1, 2:end-1]
		count = dot(view(A,i-b:i+b), kern)
		out[i] = liveordie(count, A[i])
	end
	out
end


# ╔═╡ bbbaea16-9d92-46b8-ae06-530cc3c18530
convGOL = convGOL_conv


# ╔═╡ 1df253df-6e41-4581-9d21-326924e13c86
function GOLsteps(state::AbstractMatrix, kern::AbstractMatrix, ngens::Integer)
	for _ in 1:ngens
		state = convGOL(state, kern)
	end
	state
end

# ╔═╡ 3f533909-f80d-40a4-a0ef-f5151590928a
function convolveimgs(images, kern::AbstractMatrix, ngens::Integer)
    num_imgs = size(images)[1]
    out = similar(images)
    for img_num in 1:num_imgs
       #out[img_num] = conv(images[img_num], kern)
       out[img_num] = GOLsteps(images[img_num], kern, ngens)
    end
    out
end

# ╔═╡ 15ec582b-044f-4c50-aefc-c0f9125a709f
function convolveimgs_t(images, kern::AbstractMatrix, ngens::Integer)
    num_imgs = size(images)[1]
    out = similar(images)
    Threads.@threads for img_num in 1:num_imgs
        #out[img_num] = conv(images[img_num], kern)
        out[img_num] = GOLsteps(images[img_num], kern, ngens)
    end
    out
end

# ╔═╡ ad012854-ad95-4549-8f54-f4c5f7a2604d
function tilerand(img, num::Integer) #will return a matrix like img
	outarray = Array{typeof(img)}(undef, num)
	tmp = [] #Array{Float32,2}(undef, num)
	for i in 1:num
		push!(tmp, eltype(img).(map( x-> round.(x), rand(Float32, size(img)))))
		outarray[i] = eltype(img).(map( x-> round.(x), rand(Float32, size(img))))
	end
	#eltype(img).(map(x -> round.(x), tmp))
	outarray
end	


# ╔═╡ 25a620b2-c142-4e01-a967-dd540993a322
kernel =       UInt8.([ 1 1 1   #kernel for GoL
                        1 0 1
                        1 1 1 ])



# ╔═╡ ba8f43e1-67cc-4726-90d6-1383835d9aaa
canvas_loaf = tilerand(lisa_loaf[1], batch_size)

# ╔═╡ 70f6e12c-51f8-454b-9b0a-97dc16dc368f
Gray.(canvas_loaf[1])

# ╔═╡ fe58dc89-fbee-4e74-97e8-bb2cd1b6d97b
#Now we need the mutator tensor (terminology from the article)
#Same shape as canvas_loaf, but all zeros execpt for one 1 in each "slice"
function mutate(imgs, kw=0, max_mutations = MAX_MUTATIONS)
	num_imgs = size(imgs)[1]
	img_size = size(imgs[1])
	outarray = Array{typeof(imgs[1])}(undef, num_imgs)

	for r in 1:num_imgs
		num_mutations = rand(1:max_mutations)
	    mutator = zeros(UInt8, img_size)
	    for i in 1:num_mutations #try 3 random pixels (TODO make this paramatizable)
	       x,y = rand(1:img_size[1]-kw), rand(1:img_size[2]-kw)
	       mutator[x,y] = 0x01
		end
	    outarray[r] = mutator
    end
	outarray
end


# ╔═╡ f6b0dfbe-689f-428c-92c0-fbf421deaa8b
#possibly an alternative to rmse for loss
function img_diff(a::AbstractMatrix, b::AbstractMatrix)
	sum(xor.(a,b))
end


# ╔═╡ 36f8fdea-ea63-4ab0-820a-072049e1fe82
function hill_climb(original, canvas, iterations, kern=kernel, num_gens=n_generations)
	best_score  = Inf
	best_canvas = copy(canvas)
	s_canvas    = copy(canvas)
	fitness_progress = []
	for run in 1:iterations
		#s_canvas = xor.(s_canvas, mutate(canvas, size(kernel)[1]))
		s_canvas = [xor.(a,b) for (a,b) in zip(s_canvas, mutate(canvas, size(kern)[1]))]
		#saved_canvas = copy(s_canvas)
		#convGOL_all!(s_canvas, kern, num_gens) #s_canvas will be modified
        after_canvas = convolveimgs_t(s_canvas, kern, num_gens)

		#rmse_vals = rmse.(original, s_canvas)
		rmse_vals = img_diff.(original, after_canvas)
		curr_min  = minimum(rmse_vals)
		push!(fitness_progress, curr_min) 
		if curr_min < best_score
			best_score = curr_min
			best_canvas = tileimg(s_canvas[argmin(rmse_vals)], size(s_canvas)[1])
		end
		s_canvas = best_canvas
	end
	return s_canvas, fitness_progress
end


# ╔═╡ 7ba5024f-1d53-4571-90e3-fe73bb5c110a
result, fitnesses = hill_climb(lisa_loaf, canvas_loaf, 20)



# ╔═╡ a3f2bb8c-8dcf-449e-9fc4-758413404728
plot(fitnesses)

# ╔═╡ 55577b04-f510-4ade-b557-99da78c71de6
Gray.(result[1])

# ╔═╡ 1845b690-0202-4d71-8958-e5771c1d52ec
result1, fitnesses1 = hill_climb(lisa_loaf, canvas_loaf, 200)


# ╔═╡ 3fb3b9bb-d34d-4aa3-b2ed-5a135b8ea9a6
plot(fitnesses1)

# ╔═╡ e0512b7c-4e97-49a4-ab9b-2741c62b328e
with_terminal() do
	@btime convolveimgs_t(canvas_loaf, kernel, n_generations)
end

# ╔═╡ db8f5517-a0fa-49f7-b855-08bd291f2dd2
with_terminal() do
	@btime convolveimgs(canvas_loaf, kernel, n_generations)
end

# ╔═╡ 09abcf0e-8a1e-49ba-8725-e6ead7ae0abf
result2, fitnesses2 = hill_climb(lisa_loaf, canvas_loaf, 2000)


# ╔═╡ 7fc48ec0-cbdc-485d-afd9-c36c46142a6f
plot(fitnesses2)

# ╔═╡ 09371cc5-fc42-4a26-a882-a6256c6fcb1d
Gray.(GOLsteps(result2[1], kernel, n_generations))

# ╔═╡ Cell order:
# ╠═0b6609a2-968c-11eb-3684-ad836c52007b
# ╠═d12c1e0d-6663-4372-ae5a-9a7e709ee518
# ╠═0916b0a7-fe88-4d23-94c6-0132dedee62f
# ╠═95de85db-1bf3-43e5-978f-97c736853ef7
# ╠═e0a49733-2b1f-4aed-ae7b-5d33b7aa6e99
# ╠═950c80a8-0860-4c9d-9081-218b8e88f3c5
# ╠═ab895b8d-028e-4091-820f-93bce888b3ba
# ╠═f8386e21-3d9c-4363-8352-e90858b1a47d
# ╠═1b348db5-6185-4c8a-a6d5-979fa03f9f61
# ╠═bbbaea16-9d92-46b8-ae06-530cc3c18530
# ╠═1df253df-6e41-4581-9d21-326924e13c86
# ╠═3f533909-f80d-40a4-a0ef-f5151590928a
# ╠═15ec582b-044f-4c50-aefc-c0f9125a709f
# ╠═ad012854-ad95-4549-8f54-f4c5f7a2604d
# ╠═25a620b2-c142-4e01-a967-dd540993a322
# ╠═ba8f43e1-67cc-4726-90d6-1383835d9aaa
# ╠═70f6e12c-51f8-454b-9b0a-97dc16dc368f
# ╠═fe58dc89-fbee-4e74-97e8-bb2cd1b6d97b
# ╠═f6b0dfbe-689f-428c-92c0-fbf421deaa8b
# ╠═36f8fdea-ea63-4ab0-820a-072049e1fe82
# ╠═7ba5024f-1d53-4571-90e3-fe73bb5c110a
# ╠═a3f2bb8c-8dcf-449e-9fc4-758413404728
# ╠═55577b04-f510-4ade-b557-99da78c71de6
# ╠═1845b690-0202-4d71-8958-e5771c1d52ec
# ╠═3fb3b9bb-d34d-4aa3-b2ed-5a135b8ea9a6
# ╠═e0512b7c-4e97-49a4-ab9b-2741c62b328e
# ╠═db8f5517-a0fa-49f7-b855-08bd291f2dd2
# ╠═09abcf0e-8a1e-49ba-8725-e6ead7ae0abf
# ╠═7fc48ec0-cbdc-485d-afd9-c36c46142a6f
# ╠═09371cc5-fc42-4a26-a882-a6256c6fcb1d
