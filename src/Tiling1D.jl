module Tiling1D

const __iris_dmem_map = Dict{Any, Any}()
const __pin_map = Dict{Any, Any}()
export tile_array, Tiling

struct Tile1D{T, V<:AbstractVector{T}}
    view::SubArray{T,1,V,Tuple{UnitRange{Int},},true}
    dmem::Any
    index::Int
end

# Function to split a 1D array into tiles (segments).
#   - A: 1D array (vector)
#   - tile_length: desired length for each tile.
#   - flattened (optional): for consistency, always returns a vector of tiles.
#
# Returns a vector of Tile1D objects, each containing a view into A,
# an optional device memory pointer, and the tile's index.
function tile_array(A::AbstractVector, tile_length::Int)
    n = length(A)
    tiles = Vector{Tile1D{eltype(A), typeof(A)}}()
    idx = 0
    for i in 1:tile_length:n
        idx += 1
        end_index = min(i + tile_length - 1, n)
        view_tile = @view A[i:end_index]
        # Use the actual tile length for the device size.
        current_tile_length = end_index - i + 1
        dev_size = [current_tile_length]
        offset = [i - 1]
        mem = nothing
        if isdefined(Main, :IrisHRT)
            key = (pointer(A), offset, dev_size)
            if !haskey(__iris_dmem_map, key)
                mem = Main.IrisHRT.dmem_offset(A, dev_size, offset)
                __iris_dmem_map[key] = mem
            else
                mem = __iris_dmem_map[key]
            end
        end
        push!(tiles, Tile1D(view_tile, mem, idx))
    end
    return tiles
end

# A struct to package the tiling of a 1D array.
# Fields:
#   - A: the original 1D array
#   - tile_size, tile_cols: dimensions for each tile
#   - tiles: a vector of tile views
#   - n_tile_size, n_tile_cols: the number of tile rows and columns
struct Tiling{T, V<:AbstractVector{T}}
    A::V
    tile_size::Int
    tiles::Any
    n_tiles::Int
end

# Tiling(A, tile_size)
#    Create a Tiling object from a 1D array `A` by splitting it into tiles of size
#    `tile_size`. The tile size
#    dimensions are not multiples of the tile size.
function Tiling(A::AbstractVector, tile_size::Int)
    n = size(A, 1)
    n_tiles = cld(n, tile_size)
    tiles = tile_array(A, tile_size)
    if ! haskey(__pin_map, pointer(A))
        if isdefined(Main, :IrisHRT)
            Main.IrisHRT.register_pin(A)
            __pin_map[pointer(A)] = true
        end
    end
    return Tiling{eltype(A), typeof(A)}(A, tile_size, tiles, n_tiles)
end

end  # module Tiling1D

macro tiling1d(args...)
    # Parse the arguments: expect keyword arguments and optionally a block.
    local data_expr      = nothing
    local tile_size_expr = :(2)
    local block_expr     = nothing
    local name_expr      = :(tile)

    # Process each argument.
    for arg in args
        if arg isa Expr && arg.head == :(=)
            # Keyword argument of the form key = value.
            if arg.args[1] == :data
                data_expr = arg.args[2]
            elseif arg.args[1] == :tile_size
                tile_size_expr = arg.args[2]
            elseif arg.args[1] == :name
                name_expr = arg.args[2]
            else
                error("Unknown keyword: ", arg.args[1])
            end
        else
            # Assume this is the loop body block.
            block_expr = arg
        end
    end

    if block_expr === nothing
        return esc(quote
            local __data = $data_expr
            if __data isa Tuple
                local __all_tiling = 
                    [ Tiling1D.Tiling(x, $tile_size_expr).tiles  for x in __data ]
                local __d = [ Tuple([__all_tiling[k][j] for k in 1:length(__all_tiling)]) for j in 1:length(__all_tiling[1]) ]
                __d
            else
                local __tiling = Tiling1D.Tiling(__data, $tile_size_expr)
                __tiling.tiles
            end
        end)
    else
        return esc(quote
            # (Optional) Debug print.
            local __data = $data_expr
            if __data isa Tuple
                local __all_tiling = 
                    [ Tiling1D.Tiling(x, $tile_size_expr).tiles for x in __data ]
                local __d = [ Tuple([__all_tiling[k][j] for k in 1:length(__all_tiling)]) for j in 1:length(__all_tiling[1]) ]
                for (__index, __tile) in enumerate(__d)
                    let $name_expr = (index=__index, data=__tile)
                        $block_expr
                    end
                end
            else
                local __d = Tiling1D.Tiling(__data, $tile_size_expr)
                for (__index, __tile) in enumerate(__d.tiles)
                    let $name_expr = (index=__index, data=__tile)
                        $block_expr
                    end
                end
            end
       end)
    end
end

