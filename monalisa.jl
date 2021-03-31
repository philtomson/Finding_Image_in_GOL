### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ a3e50b96-81f2-11eb-294d-153882ea5f67
begin
	using Images
	using ImageView
	using Ditherings
	using PlutoUI
	using DSP #for conv
	using LoopVectorization, OffsetArrays
	using CUDA
	using Plots
end
	

# ╔═╡ b43bbe96-81f1-11eb-1a1f-e798274b82ac
md"Load up the dependencies"


# ╔═╡ 5411b618-8dbf-11eb-1e0d-e311384e9d9d
n_gens = 6

# ╔═╡ 3e307332-917a-11eb-321f-b3a59fab394f
MAX_MUTATIONS = 8

# ╔═╡ bdd9aba4-8771-11eb-29e7-b544121ce870
kernel = UInt8.([ 1 1 1   #kernel for GoL
                  1 0 1
                  1 1 1 ])


# ╔═╡ 8f7564fc-90e4-11eb-0508-cf1f28667f24
test_board = UInt8.([ 0 0 0 0 0 0 0 0
		              0 0 0 0 0 0 0 0
		              0 1 1 1 0 0 0 0
		              0 1 0 0 0 0 0 0 
		              0 0 1 0 0 0 0 0 
		              0 0 0 0 0 0 0 0
		              0 0 0 0 0 1 1 1
		              0 0 0 0 0 0 0 0
		             ])

# ╔═╡ 93056986-877b-11eb-294a-0b3802b5ba62
liveordie(count::Integer, current) = 
(current == 0 && count==3) || (current == 1 && (1 < count <=3))


# ╔═╡ 150916da-8772-11eb-30e5-6b2ab6de1f24
function convGOLold(A::AbstractMatrix, kern::AbstractMatrix)
	outtype = eltype(A)
	out = zeros(outtype, size(A))
	iterize = size(A) #.- size(kern)
    @inbounds for J in CartesianIndices(iterize.-(1,1)) #@avx
        count = zero(outtype)
        for I ∈ CartesianIndices(kern)
            count += A[I + J] * kern[I]
        end
		#Any live cell with two or three live neighbours survives.
        #Any dead cell with three live neighbours becomes a live cell.
        #All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        out[J] =  count #outtype(liveordie(count, A[J]))
    end
    out
end

# ╔═╡ 88fbe00c-90ea-11eb-0e6c-eb8bd504babf
function convGOL(A::AbstractMatrix, kern::AbstractMatrix)
	
	counts = conv(A, kern)[2:end-1, 2:end-1]
	out    = zeros(eltype(A), size(A))
	#size(counts) should == size(A)
	@inbounds for i in CartesianIndices(size(A))
		out[i] = liveordie(counts[i], A[i])
	end
	out
end
		
	
	
	

# ╔═╡ e0cd1dd8-91cd-11eb-2c6d-c7a93f002d8b
function GOLsteps(state::AbstractMatrix, kern::AbstractMatrix, ngens::Integer)
	for _ in 1:ngens
		state = convGOL(state, kern)
	end
	state
end

# ╔═╡ 6b369612-90e5-11eb-1914-256098e50788
test_board1 = convGOL(test_board, kernel)

# ╔═╡ 7c222bb8-90f0-11eb-277f-0d8d88b385dd
Gray.(test_board)

# ╔═╡ 7f332422-90e6-11eb-39c2-8d2397d3e2a5
Gray.(test_board1)

# ╔═╡ 342aa2b0-8d1f-11eb-315d-411b639bd426
#conv the whole "loaf"
function convGOL_all!(A::Array{Array{UInt8,2},1}, kern::AbstractMatrix, ngens::Integer)
   #mapslices( img -> convGOL(img, kern), A, (2,3))
	num_imgs = size(A)[1]
	
	    for img_num in 1:num_imgs
	        tmp = GOLsteps(A[img_num], kern, ngens)
		    A[img_num] = tmp
		end
end

# ╔═╡ f8a49e28-8830-11eb-1bc1-61485a281e82
function convGPU(AA::AbstractMatrix, k::AbstractMatrix)
    kern = CuArray(k) #  |> gpu
	A    = CuArray(AA) # |> gpu
	outtype = eltype(A)
	out = zeros(size(A))
    @inbounds for J in CartesianIndices(out) #@avx
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J] * kern[I]
        end
		#Any live cell with two or three live neighbours survives.
        #Any dead cell with three live neighbours becomes a live cell.
        #All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        out[J] =  outtype(liveordie(Integer(tmp), convert(UInt8,(A[J]))))
    end
    out
end

# ╔═╡ 3a9216ee-8c23-11eb-02c5-1ba46ec12124
function tileimg(img::AbstractArray, num::Integer)
	outarray = Array{typeof(img)}(undef, num)
	for i in 1:num
		outarray[i] = copy(img)
	end
	outarray
end

# ╔═╡ 20520da4-8dbb-11eb-3712-850c67408a26


# ╔═╡ 09742ef2-8c27-11eb-2b1d-55623b383688
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

# ╔═╡ 186c7c1c-8c2d-11eb-374e-a70f26fbf197
#Now we need the mutator tensor (terminology from the article)
#Same shape as canvas_loaf, but all zeros execpt for one 1 in each "slice"
function mutate_one(sz, kw=0)
    mutator = zeros(UInt8, sz)
    x,y = rand(1:sz[1]-kw), rand(1:sz[2]-kw)
	mutator[x,y] = 0x01
	mutator
end

# ╔═╡ 00821a34-8c2e-11eb-0e4a-09103605bb7b
#function Base.xor(a::Array{UInt8,2}, b::Array{UInt8,2})
#	xor.(a,b)
#end

# ╔═╡ 39fb2038-81f4-11eb-3b7f-d11eec36b65d
begin
   monalisa_img = mktemp() do fn,f
    download("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/483px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg", fn)
    load(fn)
   end
end


# ╔═╡ 0ad05da4-81f5-11eb-34fb-0fd4b2290487
md"Now convert to grayscale"

# ╔═╡ 157b5560-81f5-11eb-33b2-c5f0358999c5
monalisa_img_gs = Gray.(monalisa_img)


# ╔═╡ 52ad9c40-81f5-11eb-19ab-5b007947dd63
md"Now dither the grayscale image"

# ╔═╡ 6a5930b8-81f5-11eb-0dfc-ebd54f961a42
monalisa_img_dithered = Ditherings.FloydSteinbergDither4Sample(monalisa_img_gs, Ditherings.ZeroOne)

# ╔═╡ a4a26ab4-81f5-11eb-1669-4feb6706f82d
md"Convert to high contrast grayscale (0 or 1) - NOTE: output of Ditherings is RGB - must convert to Gray again"

# ╔═╡ d8433646-81f5-11eb-23f1-bb22fe75a2f9
lisa_contrast_img = map(x -> round(x), Gray.(monalisa_img_dithered))

# ╔═╡ 15d7db76-81f7-11eb-0525-478de1404593
typeof(Int32.(lisa_contrast_img))

# ╔═╡ c6441f9c-81f7-11eb-3481-aff6c9764877
begin
   md"? How could we convert to be Array{Int32,2} ?"
	lisa_img = Gray.(Int8.(lisa_contrast_img))
	typeof(lisa_img)
	#make sure there are no values other than 0 and 1 in image:
	size(filter(x-> 0.0 > x > 1.0, lisa_img))
	
end


# ╔═╡ df507380-81f7-11eb-2764-2504a167b1fc
begin
	
	batch_size = 50
	n_generations = 6
	width,height = size(lisa_img)
	#lisa_loaf = UInt8.(repeat(lisa_img, 1, 1, batch_size))
	lisa_loaf = tileimg(UInt8.(lisa_img), batch_size)
	with_terminal() do
	   @show size(lisa_loaf)
	   @show typeof(lisa_loaf[:,:,1][1,1])
	   @show lisa_loaf[:,:,1][1,1]
	end
	
end

# ╔═╡ 22d9f404-836d-11eb-3d67-0b6f64d917d4
lisa_loaf[1] #look at first image in "loaf"


# ╔═╡ b98ee864-8c23-11eb-3aac-bd475a3afb31
Gray.(lisa_loaf[1])

# ╔═╡ 46136f9c-836d-11eb-3d3e-75b1b2fa1e17
#make random canvases same size/shape as lisa_loaf
begin
	#canvas_loaf = Gray.(rand(Float32, 720,483,50))
	#canvas_loaf[:,:,1]
	
	#rand_loaf = Gray.(rand(Float32, 720, 483, 50))
	#canvas_loaf = UInt8.(map(x -> round(x), rand_loaf))
	#Gray.(canvas_loaf[:,:,1])
	
	canvas_loaf = tilerand(lisa_loaf[1], batch_size)
	typeof(lisa_loaf[1])
	#ray.(canvas_loaf[1])
	
	
end

# ╔═╡ b8b556dc-91d3-11eb-1885-e3e47450ab87
Gray.(canvas_loaf[1])

# ╔═╡ 33adbf88-8c2e-11eb-16fc-0baa78a3a1eb
size(canvas_loaf)

# ╔═╡ eb030774-8861-11eb-1710-c573b19b5b52
typeof(canvas_loaf)

# ╔═╡ 07289ae2-8862-11eb-3c83-f97cb1a1261f
typeof(lisa_loaf)

# ╔═╡ c27b583a-837d-11eb-0e68-e1d2bd95cc5e
rmse_vals = rmse.(canvas_loaf, lisa_loaf)

# ╔═╡ a1bcff44-8dbf-11eb-19c0-b5903843c494
min_rmse_val = minimum(rmse_vals)

# ╔═╡ eec7000c-8d18-11eb-0d21-571e0939ae07
argmin(rmse_vals)

# ╔═╡ cb9803a4-8d21-11eb-21ba-f734c88bcd50
canvas_loaf_before = copy(canvas_loaf)

# ╔═╡ 1e3aca6c-8d23-11eb-11f2-df2a5970a8cb
typeof(canvas_loaf_before)

# ╔═╡ 44ce58e2-8d23-11eb-2834-e91544367f04
sum.(canvas_loaf_before)

# ╔═╡ 02f86868-8d23-11eb-3f43-512eabce1c50
Gray.(canvas_loaf_before[1])

# ╔═╡ 0f31b2a0-8d1f-11eb-2113-510bc2e43e46
begin
	canvas_loaf_copy = copy(canvas_loaf)
   convGOL_all!(canvas_loaf_copy, kernel, 5)
end

# ╔═╡ 6f5ae3b4-8d23-11eb-1e2d-891a32cfe9e8
sum.(canvas_loaf)

# ╔═╡ c821cad2-8d21-11eb-0576-618733457991
rmse(canvas_loaf_before[1], canvas_loaf[1]) #WNY is this 0.0?!!

# ╔═╡ e516ec20-8d22-11eb-3e85-d5fa4d9a062f
Gray.(canvas_loaf[1])

# ╔═╡ ba00793a-8d17-11eb-2f9d-ef2ce9565851
rmse(canvas_loaf[46], lisa_loaf[46])

# ╔═╡ 31c2c444-8d18-11eb-2820-91a3e1db3965
Gray.(canvas_loaf[46])

# ╔═╡ 3dea31aa-8d18-11eb-1d81-2f102182973f
Gray.(lisa_loaf[46])

# ╔═╡ a4ed1a44-877e-11eb-2fe9-b33ca0b1f39e
ns1 = convGOL(lisa_loaf[1], kernel)

# ╔═╡ 9c5aeeb6-8861-11eb-160a-b1d0222010fb
eltype(ns1)

# ╔═╡ 8a34ff4e-8861-11eb-3a90-db8e79c3b476
eltype(lisa_loaf[:,:,1])

# ╔═╡ 40cba2f0-8798-11eb-259f-cf57c4f84db5
Gray.(ns1)

# ╔═╡ 9575a962-885f-11eb-2a98-5be3d7e2d906
#for cnvs in canvas_loaf
#	imshow(Gray.(cnvs))
#end
Gray.(canvas_loaf[1])

# ╔═╡ fec3b1fc-885f-11eb-347d-99257f131975
ns2 = convGOL(canvas_loaf[50], kernel)

# ╔═╡ 2f717956-8860-11eb-2120-d1f872b570ec
typeof(ns2)

# ╔═╡ 8fce7072-8862-11eb-2704-e7de81aa35cb
Gray.(ns2)

# ╔═╡ 16124688-8861-11eb-26d8-1b28de31c603
ns3 = convGOL(ns2, kernel)

# ╔═╡ 8481b3fa-8862-11eb-0faf-1be9030dba9d
Gray.(ns3)

# ╔═╡ 5de20bc6-8cf6-11eb-3e64-fd4356d32bf0
typeof(canvas_loaf[1])


# ╔═╡ 763f5126-8b73-11eb-0ba4-651a2b1cc8c6
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

# ╔═╡ 69bc714e-8b74-11eb-2956-f106bef8bb3a
mutator = mutate(canvas_loaf, size(kernel)[1])


# ╔═╡ 2520cf00-8b7e-11eb-2e97-3d1a4d78e432

size(mutator) #[699, 480] = 0x01


# ╔═╡ b0a747bc-8cf7-11eb-3637-51d2ae76e71f
Gray.(mutator[46])

# ╔═╡ 03144d4c-8dc5-11eb-07ea-91afc1e4d3ad
typeof(canvas_loaf)

# ╔═╡ bedeb286-8dbd-11eb-024c-bd1c613d0792
function hill_climb(original, canvas, iterations, kernel=kernel, num_gens=n_gens)
	best_score  = Inf
	best_canvas = copy(canvas)
	s_canvas    = copy(canvas)
	fitness_progress = []
	for run in 1:iterations
		#s_canvas = xor.(s_canvas, mutate(canvas, size(kernel)[1]))
		s_canvas = [xor.(a,b) for (a,b) in zip(s_canvas, mutate(canvas, size(kernel)[1]))]
		convGOL_all!(s_canvas, kernel, num_gens)
		rmse_vals = rmse.(original, s_canvas)
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

# ╔═╡ 867e347e-8dbe-11eb-10b9-7db60f1053d8
result, fitnesses = hill_climb(lisa_loaf, canvas_loaf, 250)

# ╔═╡ b5a98880-9196-11eb-179d-71191f142ccf
plot(fitnesses)

# ╔═╡ 19bda112-91d3-11eb-0b19-65c359b05954
fitnesses


# ╔═╡ 68739244-8dc8-11eb-3fc0-9fd199e0ec4a
Gray.(result[1])

# ╔═╡ 56286b72-9177-11eb-38db-afed9d718b63
result2, fitnesses2 = hill_climb(lisa_loaf, result, 500)

# ╔═╡ cdeb1670-9196-11eb-2419-798a88c6c87d
plot(fitnesses2)

# ╔═╡ 04f3a868-9178-11eb-0466-8d60e69eabab
Gray.(result2[1])

# ╔═╡ 0b5f2110-91b3-11eb-17be-3b18496ad658
Gray.(GOLsteps(result2[1], kernel, 1000))

# ╔═╡ 164cd10a-9178-11eb-28b9-ddfb5aec1f2d
Gray.(xor.(result2[1], result[1]))

# ╔═╡ 552ed396-9178-11eb-2064-dd0a1cddb31c
rmse(result[1], result2[1])

# ╔═╡ 163b801e-9178-11eb-0d24-31b50077a4d6


# ╔═╡ abca9ec8-9108-11eb-1f9d-89c1a17b7576
Gray.(canvas_loaf[1])

# ╔═╡ fedd0dac-8dc6-11eb-3594-cd1ab3de9805
#find the best one:
rmse_vals1 = rmse.(result2, lisa_loaf)

# ╔═╡ 8995305e-8dc8-11eb-362e-81fcdf7ed4cf
minimum(rmse_vals1)

# ╔═╡ bfdb81c2-8dc8-11eb-353d-71df40b46b1b
argmin(rmse_vals1)

# ╔═╡ a017b720-8dc8-11eb-3352-375f41ea9de9
Gray.(result[argmin(rmse_vals1)])

# ╔═╡ a00c0614-8dc8-11eb-1c36-1d38506baff6


# ╔═╡ 5ba08d2c-8da4-11eb-18f0-11d1a6d22219
modulod = (canvas_loaf[46] .+ mutator[46]) .%0x02

# ╔═╡ Cell order:
# ╠═b43bbe96-81f1-11eb-1a1f-e798274b82ac
# ╠═a3e50b96-81f2-11eb-294d-153882ea5f67
# ╠═5411b618-8dbf-11eb-1e0d-e311384e9d9d
# ╠═3e307332-917a-11eb-321f-b3a59fab394f
# ╠═bdd9aba4-8771-11eb-29e7-b544121ce870
# ╠═8f7564fc-90e4-11eb-0508-cf1f28667f24
# ╠═93056986-877b-11eb-294a-0b3802b5ba62
# ╠═150916da-8772-11eb-30e5-6b2ab6de1f24
# ╠═88fbe00c-90ea-11eb-0e6c-eb8bd504babf
# ╠═e0cd1dd8-91cd-11eb-2c6d-c7a93f002d8b
# ╠═6b369612-90e5-11eb-1914-256098e50788
# ╠═7c222bb8-90f0-11eb-277f-0d8d88b385dd
# ╠═7f332422-90e6-11eb-39c2-8d2397d3e2a5
# ╠═342aa2b0-8d1f-11eb-315d-411b639bd426
# ╠═f8a49e28-8830-11eb-1bc1-61485a281e82
# ╠═3a9216ee-8c23-11eb-02c5-1ba46ec12124
# ╠═20520da4-8dbb-11eb-3712-850c67408a26
# ╠═09742ef2-8c27-11eb-2b1d-55623b383688
# ╠═186c7c1c-8c2d-11eb-374e-a70f26fbf197
# ╠═00821a34-8c2e-11eb-0e4a-09103605bb7b
# ╠═39fb2038-81f4-11eb-3b7f-d11eec36b65d
# ╠═0ad05da4-81f5-11eb-34fb-0fd4b2290487
# ╠═157b5560-81f5-11eb-33b2-c5f0358999c5
# ╠═52ad9c40-81f5-11eb-19ab-5b007947dd63
# ╠═6a5930b8-81f5-11eb-0dfc-ebd54f961a42
# ╠═a4a26ab4-81f5-11eb-1669-4feb6706f82d
# ╠═d8433646-81f5-11eb-23f1-bb22fe75a2f9
# ╠═15d7db76-81f7-11eb-0525-478de1404593
# ╠═c6441f9c-81f7-11eb-3481-aff6c9764877
# ╠═df507380-81f7-11eb-2764-2504a167b1fc
# ╠═22d9f404-836d-11eb-3d67-0b6f64d917d4
# ╠═b98ee864-8c23-11eb-3aac-bd475a3afb31
# ╠═46136f9c-836d-11eb-3d3e-75b1b2fa1e17
# ╠═b8b556dc-91d3-11eb-1885-e3e47450ab87
# ╠═33adbf88-8c2e-11eb-16fc-0baa78a3a1eb
# ╠═eb030774-8861-11eb-1710-c573b19b5b52
# ╠═07289ae2-8862-11eb-3c83-f97cb1a1261f
# ╠═c27b583a-837d-11eb-0e68-e1d2bd95cc5e
# ╠═a1bcff44-8dbf-11eb-19c0-b5903843c494
# ╠═eec7000c-8d18-11eb-0d21-571e0939ae07
# ╠═cb9803a4-8d21-11eb-21ba-f734c88bcd50
# ╠═1e3aca6c-8d23-11eb-11f2-df2a5970a8cb
# ╠═44ce58e2-8d23-11eb-2834-e91544367f04
# ╠═02f86868-8d23-11eb-3f43-512eabce1c50
# ╠═0f31b2a0-8d1f-11eb-2113-510bc2e43e46
# ╠═6f5ae3b4-8d23-11eb-1e2d-891a32cfe9e8
# ╠═c821cad2-8d21-11eb-0576-618733457991
# ╠═e516ec20-8d22-11eb-3e85-d5fa4d9a062f
# ╠═ba00793a-8d17-11eb-2f9d-ef2ce9565851
# ╠═31c2c444-8d18-11eb-2820-91a3e1db3965
# ╠═3dea31aa-8d18-11eb-1d81-2f102182973f
# ╠═a4ed1a44-877e-11eb-2fe9-b33ca0b1f39e
# ╠═9c5aeeb6-8861-11eb-160a-b1d0222010fb
# ╠═8a34ff4e-8861-11eb-3a90-db8e79c3b476
# ╠═40cba2f0-8798-11eb-259f-cf57c4f84db5
# ╠═9575a962-885f-11eb-2a98-5be3d7e2d906
# ╠═fec3b1fc-885f-11eb-347d-99257f131975
# ╠═2f717956-8860-11eb-2120-d1f872b570ec
# ╠═8fce7072-8862-11eb-2704-e7de81aa35cb
# ╠═16124688-8861-11eb-26d8-1b28de31c603
# ╠═8481b3fa-8862-11eb-0faf-1be9030dba9d
# ╠═5de20bc6-8cf6-11eb-3e64-fd4356d32bf0
# ╠═763f5126-8b73-11eb-0ba4-651a2b1cc8c6
# ╠═69bc714e-8b74-11eb-2956-f106bef8bb3a
# ╠═2520cf00-8b7e-11eb-2e97-3d1a4d78e432
# ╠═b0a747bc-8cf7-11eb-3637-51d2ae76e71f
# ╠═03144d4c-8dc5-11eb-07ea-91afc1e4d3ad
# ╠═bedeb286-8dbd-11eb-024c-bd1c613d0792
# ╠═867e347e-8dbe-11eb-10b9-7db60f1053d8
# ╠═b5a98880-9196-11eb-179d-71191f142ccf
# ╠═19bda112-91d3-11eb-0b19-65c359b05954
# ╠═68739244-8dc8-11eb-3fc0-9fd199e0ec4a
# ╠═56286b72-9177-11eb-38db-afed9d718b63
# ╠═cdeb1670-9196-11eb-2419-798a88c6c87d
# ╠═04f3a868-9178-11eb-0466-8d60e69eabab
# ╠═0b5f2110-91b3-11eb-17be-3b18496ad658
# ╠═164cd10a-9178-11eb-28b9-ddfb5aec1f2d
# ╠═552ed396-9178-11eb-2064-dd0a1cddb31c
# ╠═163b801e-9178-11eb-0d24-31b50077a4d6
# ╠═abca9ec8-9108-11eb-1f9d-89c1a17b7576
# ╠═fedd0dac-8dc6-11eb-3594-cd1ab3de9805
# ╠═8995305e-8dc8-11eb-362e-81fcdf7ed4cf
# ╠═bfdb81c2-8dc8-11eb-353d-71df40b46b1b
# ╠═a017b720-8dc8-11eb-3352-375f41ea9de9
# ╠═a00c0614-8dc8-11eb-1c36-1d38506baff6
# ╠═5ba08d2c-8da4-11eb-18f0-11d1a6d22219
