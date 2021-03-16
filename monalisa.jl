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
end
	

# ╔═╡ b43bbe96-81f1-11eb-1a1f-e798274b82ac
md"Load up the dependencies"


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
	lisa_loaf = repeat(lisa_img, 1, 1, batch_size)
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
	canvas_loaf = Gray.(rand(Float32, 720,483,50))
	canvas_loaf[:,:,1]
end

# ╔═╡ c27b583a-837d-11eb-0e68-e1d2bd95cc5e
rmse(canvas_loaf, lisa_loaf)

# ╔═╡ Cell order:
# ╠═b43bbe96-81f1-11eb-1a1f-e798274b82ac
# ╠═a3e50b96-81f2-11eb-294d-153882ea5f67
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
# ╠═c27b583a-837d-11eb-0e68-e1d2bd95cc5e
