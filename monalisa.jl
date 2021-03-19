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
end
	

# ╔═╡ b43bbe96-81f1-11eb-1a1f-e798274b82ac
md"Load up the dependencies"


# ╔═╡ bdd9aba4-8771-11eb-29e7-b544121ce870
kernel = [ 1.0 1.0 1.0
              1.0 0.0 1.0
              1.0 1.0 1.0 ]


# ╔═╡ 93056986-877b-11eb-294a-0b3802b5ba62
liveordie(count::Integer, current) = 
(current == 0 && count==3) || (current == 1 && (1 < count <=3))


# ╔═╡ 150916da-8772-11eb-30e5-6b2ab6de1f24
function convGOL(A::AbstractMatrix, kern::AbstractMatrix)
	outtype = eltype(A)
	out = zeros(outtype, size(A))
    @inbounds for J in CartesianIndices(out) #@avx
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J] * kern[I]
        end
		#Any live cell with two or three live neighbours survives.
        #Any dead cell with three live neighbours becomes a live cell.
        #All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        out[J] =  outtype(liveordie(Integer(tmp), A[J]))
    end
    out
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
	lisa_loaf = UInt8.(repeat(lisa_img, 1, 1, batch_size))
	with_terminal() do
	   @show size(lisa_loaf)
	   @show typeof(lisa_loaf[:,:,1][1,1])
	   @show lisa_loaf[:,:,1][1,1]
	end
	
end

# ╔═╡ 22d9f404-836d-11eb-3d67-0b6f64d917d4
lisa_loaf[:,:, 1] #look at first image in "loaf"


# ╔═╡ 46136f9c-836d-11eb-3d3e-75b1b2fa1e17
#make random canvases same size/shape as lisa_loaf
begin
	#canvas_loaf = Gray.(rand(Float32, 720,483,50))
	#canvas_loaf[:,:,1]
	
	rand_loaf = Gray.(rand(Float32, 720, 483, 50))
	canvas_loaf = UInt8.(map(x -> round(x), rand_loaf))
	Gray.(canvas_loaf[:,:,1])
end

# ╔═╡ eb030774-8861-11eb-1710-c573b19b5b52
typeof(canvas_loaf)

# ╔═╡ 07289ae2-8862-11eb-3c83-f97cb1a1261f
typeof(lisa_loaf)

# ╔═╡ c27b583a-837d-11eb-0e68-e1d2bd95cc5e
rmse(canvas_loaf, lisa_loaf)

# ╔═╡ 81af2678-8773-11eb-2dbe-8f833fa742b9
lisa_loaf[:,:,1][230,230] == 1



# ╔═╡ ebe1b162-877c-11eb-05ec-f3e4400ac8c7
convert(UInt8, lisa_loaf[:,:,1][230,230]) == 1

# ╔═╡ 5c5c6f2c-8782-11eb-34f7-3beb3a7df00a
typeof(UInt8.(lisa_loaf[:,:,1]))

# ╔═╡ a4ed1a44-877e-11eb-2fe9-b33ca0b1f39e
ns1 = convGOL(lisa_loaf[:,:,1], kernel)

# ╔═╡ 9c5aeeb6-8861-11eb-160a-b1d0222010fb
eltype(ns1)

# ╔═╡ 8a34ff4e-8861-11eb-3a90-db8e79c3b476
eltype(lisa_loaf[:,:,1])

# ╔═╡ 40cba2f0-8798-11eb-259f-cf57c4f84db5
Gray.(ns1)

# ╔═╡ 9575a962-885f-11eb-2a98-5be3d7e2d906
for r in 1:size(canvas_loaf,3)
	Gray.(canvas_loaf[:,:,r])
end

# ╔═╡ fec3b1fc-885f-11eb-347d-99257f131975
ns2 = convGOL(canvas_loaf[:,:,1], kernel)

# ╔═╡ 2f717956-8860-11eb-2120-d1f872b570ec
typeof(ns2)

# ╔═╡ 8fce7072-8862-11eb-2704-e7de81aa35cb
Gray.(ns2)

# ╔═╡ 16124688-8861-11eb-26d8-1b28de31c603
ns3 = convGOL(ns2, kernel)

# ╔═╡ 8481b3fa-8862-11eb-0faf-1be9030dba9d
Gray.(ns3)

# ╔═╡ Cell order:
# ╠═b43bbe96-81f1-11eb-1a1f-e798274b82ac
# ╠═a3e50b96-81f2-11eb-294d-153882ea5f67
# ╠═bdd9aba4-8771-11eb-29e7-b544121ce870
# ╠═93056986-877b-11eb-294a-0b3802b5ba62
# ╠═150916da-8772-11eb-30e5-6b2ab6de1f24
# ╠═f8a49e28-8830-11eb-1bc1-61485a281e82
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
# ╠═46136f9c-836d-11eb-3d3e-75b1b2fa1e17
# ╠═eb030774-8861-11eb-1710-c573b19b5b52
# ╠═07289ae2-8862-11eb-3c83-f97cb1a1261f
# ╠═c27b583a-837d-11eb-0e68-e1d2bd95cc5e
# ╠═81af2678-8773-11eb-2dbe-8f833fa742b9
# ╠═ebe1b162-877c-11eb-05ec-f3e4400ac8c7
# ╠═5c5c6f2c-8782-11eb-34f7-3beb3a7df00a
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
