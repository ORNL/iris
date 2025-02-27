module Tiling2D

const __iris_dmem_map = Dict{Any, Any}()
const __pin_map = Dict{Any, Any}()
export tile_array, tile_row_iterator, tile_col_iterator, Tiling, row_tiles, col_tiles, tile_pointer

struct Tile2D{T, M<:AbstractMatrix{T}}
    view::SubArray{T,2,M,Tuple{UnitRange{Int},UnitRange{Int}},false}
    dmem::Any
    row_index::Int
    col_index::Int
end

# Function to split a 2D array into tiles.
# tile_array(A, tile_rows, tile_cols)
#   - A: 2D array (matrix)
#   - tile_rows, tile_cols: dimensions of each tile.
# Returns a vector of views (subarrays) into A.
function tile_array(A::AbstractMatrix, tile_rows::Int, tile_cols::Int, flattened=true::Bool)
    nrows, ncols = size(A)
    tiles = Vector{Tile2D{eltype(A), typeof(A)}}()
    row_index = 0
    col_index = 0
    for i in 1:tile_rows:nrows
        row_index += 1
        col_index = 0
        for j in 1:tile_cols:ncols
            col_index += 1
            row_end = min(i + tile_rows - 1, nrows)
            col_end = min(j + tile_cols - 1, ncols)
            view_tile = @view A[i:row_end, j:col_end]
            dev_size = [tile_cols, tile_rows]
            offset = [j-1, i-1]
            ptr = pointer(view_tile)
            mem = nothing
            if isdefined(Main, :IrisHRT)
                key = (pointer(A), offset, dev_size)
                if ! haskey(__iris_dmem_map, key)
                    mem = Main.IrisHRT.dmem_offset(A, dev_size, offset)
                    __iris_dmem_map[key] = mem
                else
                    mem = __iris_dmem_map[key]
                end
            end
            push!(tiles, Tile2D(view_tile, mem, row_index, col_index))
        end
    end
    if flattened
        return tiles
    else
        return reshape(tiles, row_index, col_index)
    end
end

# Iterators for the tiles.
# Given a vector of tiles and the number of tiles per row (num_tile_cols),
# these functions return iterators that yield a row or a column (as a vector) of tiles.
function tile_row_iterator(tiles::Vector, num_tile_cols::Int)
    nrows = length(tiles) ÷ num_tile_cols
    return ( tiles[(r-1)*num_tile_cols+1 : r*num_tile_cols] for r in 1:nrows )
end

function tile_col_iterator(tiles::Vector, num_tile_cols::Int)
    nrows = length(tiles) ÷ num_tile_cols
    return ( [ tiles[(r-1)*num_tile_cols+c] for r in 1:nrows ] for c in 1:num_tile_cols )
end

# A struct to package the tiling of a 2D array.
# Fields:
#   - A: the original 2D array
#   - tile_rows, tile_cols: dimensions for each tile
#   - tiles: a vector of tile views
#   - n_tile_rows, n_tile_cols: the number of tile rows and columns
struct Tiling{T, M<:AbstractMatrix{T}}
    A::M
    tile_rows::Int
    tile_cols::Int
    tiles::Any
    n_tile_rows::Int
    n_tile_cols::Int
    flattened::Any
    order::Any
end

# Tiling(A, tile_rows, tile_cols)
#    Create a Tiling object from a 2D array `A` by splitting it into tiles of size
#    `tile_rows`×`tile_cols`. The last tiles in a row or column may be smaller if A’s
#    dimensions are not multiples of the tile size.
function Tiling(A::AbstractMatrix, tile_rows::Int, tile_cols::Int, flattened=true::Bool, order=:row_major::Any)
    nrows, ncols = size(A)
    n_tile_rows = cld(nrows, tile_rows)
    n_tile_cols = cld(ncols, tile_cols)
    tiles = tile_array(A, tile_rows, tile_cols, false)
    if ! haskey(__pin_map, pointer(A))
        if isdefined(Main, :IrisHRT)
            Main.IrisHRT.register_pin(A)
            __pin_map[pointer(A)] = true
        end
    end
    if flattened
        if order == :row_major
            tiles = vcat(collect(eachcol(tiles))...)
            return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols, flattened, order)
        elseif order == :col_major
            tiles = vcat(collect(eachrow(tiles))...)
            return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols, flattened, order)
        end
    else
        if order == :row_major
            tiles = hcat(collect(eachrow(tiles))...)
            return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols, flattened, order)
        elseif order == :col_major
            tiles = hcat(collect(eachcol(tiles))...)
            return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols, flattened, order)
        end
    end
    return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols, flattened, order)
end

#    row_tiles(t::Tiling)
# Return an iterator over the rows of tiles. Each element is a vector containing
# the tiles in one row.
function row_tiles(t::Tiling)
    return tile_row_iterator(t.tiles, t.n_tile_cols)
end

#    col_tiles(t::Tiling)
# Return an iterator over the columns of tiles. Each element is a vector containing
# the tiles in one column.
function col_tiles(t::Tiling)
    return tile_col_iterator(t.tiles, t.n_tile_cols)
end

# Helper functions to obtain a pointer for a contiguous tile.
# Unwrap a reshaped array to get the underlying contiguous array.
function underlying_array(A::AbstractArray)
    if A isa Base.ReshapedArray
            return underlying_array(A.parent)
    elseif A isa UnitRange
            # Convert the UnitRange to an Array
            return collect(A)
    else
            return A
    end
end

# Compute a pointer to the first element of a tile.
function tile_pointer(tile::Tile2D)
    # Get the first index in each dimension for the tile's view.
    inds = ntuple(i -> first(tile.view.indices[i]), ndims(tile.view))
    # Compute the corresponding linear index in the tile's parent array.
    li = LinearIndices(tile.view.parent)[inds...]
    # Unwrap the parent to ensure we have an array that supports pointer.
    uA = underlying_array(tile.view.parent)
    return pointer(uA, li)
end

end  # module Tiling2D

"""
@tiling2d data=<A or (A, B, ...)> iterator=<:row_major|:col_major> [tile_rows=<n>] [tile_cols=<m>] [mode=<:iterate|:list>] begin
<body>
end

When mode is set to `:iterate` (the default), the macro performs the iteration over the
tiling(s) and executes the provided block for each group of corresponding tiles.
When mode is set to `:list`, the macro returns a tuple of iterators (one for each input array).

Keyword arguments:
- `data`: a 2D array or a tuple of 2D arrays.
- `iterator`: either `:row_major` or `:col_major`.
- `tile_rows` (optional): number of rows per tile (default is 64).
- `tile_cols` (optional): number of columns per tile (default is 64).
- `mode` (optional): either `:iterate` (default) or `:list`.

Inside the loop (in iteration mode), the variable `tile` is bound to:
- a single Tile2D (if one array was provided), or
- a tuple of Tile2D objects (if several arrays were provided).
"""

    
macro tiling2d(args...)
    # Parse the arguments: expect keyword arguments and optionally a block.
    local data_expr      = nothing
    local order_expr  = nothing
    local flattened = true
    local grouped = false
    local tile_rows_expr = :(2)
    local tile_cols_expr = :(2)
    local block_expr     = nothing
    local name_expr      = :(tile)

    # Process each argument.
    for arg in args
        if arg isa Expr && arg.head == :(=)
            # Keyword argument of the form key = value.
            if arg.args[1] == :data
                data_expr = arg.args[2]
            elseif arg.args[1] == :order
                order_expr = arg.args[2]
            elseif arg.args[1] == :tile_rows
                tile_rows_expr = arg.args[2]
            elseif arg.args[1] == :tile_cols
                tile_cols_expr = arg.args[2]
            elseif arg.args[1] == :flattened
                flattened = arg.args[2]
            elseif arg.args[1] == :grouped
                grouped = arg.args[2]
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
    if (grouped)
        flattened = false
    end

    if block_expr === nothing
        return esc(quote
            local __data = $data_expr
            if __data isa Tuple
                local __all_tiling = 
                    if $order_expr isa Tuple
                        [ Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr, $flattened, order_x).tiles for (x, order_x) in zip(__data, $order_expr)  ]
                    else
                        [ Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr, $flattened, $order_expr).tiles  for x in __data ]
                    end
                if $flattened
                    local __d = [ Tuple([__all_tiling[k][j] for k in 1:length(__all_tiling)]) for j in 1:length(__all_tiling[1]) ]
                    __d
                else
                    local __d = [[ Tuple([__all_tiling[k][j,i] for k in 1:length(__all_tiling)]) for j in 1:size(__all_tiling[1],2) ] for i in 1:size(__all_tiling[1],1)]
                    cat(__d..., dims=2)
                end
            else
                local __tiling = Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr, $flattened, $order_expr)
                __tiling.tiles
            end
        end)
    else
        return esc(quote
            # (Optional) Debug print.
            local __data = $data_expr
            if __data isa Tuple
                local __all_tiling = 
                    if $order_expr isa Tuple
                        [ Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr, $flattened, order_x).tiles for (x, order_x) in zip(__data, $order_expr)  ]
                    else
                        [ Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr, $flattened, $order_expr).tiles for x in __data ]
                    end  
                if $flattened
                    local __d = [ Tuple([__all_tiling[k][j] for k in 1:length(__all_tiling)]) for j in 1:length(__all_tiling[1]) ]
                    for (__index, __tile) in enumerate(__d)
                        let $name_expr = (index=__index, data=__tile)
                            $block_expr
                        end
                    end
                else
                    local __d = [[ Tuple([__all_tiling[k][j,i] for k in 1:length(__all_tiling)]) for j in 1:size(__all_tiling[1],2) ] for i in 1:size(__all_tiling[1],1)]
                    local __d2 = cat(__d..., dims=2)
                    if $grouped
                        for (__index, __group) in enumerate(eachrow(__d2))
                            let $name_expr = (index=__index, data=__group)
                                $block_expr
                            end
                        end
                    else
                        local __d3 = collect(eachrow(__d2))
                        for (__index, __rows) in enumerate(__d3)
                            let $name_expr = (index=__index, data=__rows)
                                $block_expr
                            end
                        end
                    end
                end
            else
                local __d = Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr, $flattened, $order_expr)
                if $flattened
                    for (__index, __tile) in enumerate(__d.tiles)
                        let $name_expr = (index=__index, data=__tile)
                            $block_expr
                        end
                    end
                else
                    if $grouped
                        for (__index, __group) in enumerate(eachcol(__d.tiles))
                            let $name_expr = (index=__index, data=__group)
                                $block_expr
                            end
                        end
                    else
                        for (__index, __rows) in enumerate(__d.tiles)
                            let $name_expr = (index=__index, data=__rows)
                                $block_expr
                            end
                        end
                    end
                end
            end
       end)
    end
end
