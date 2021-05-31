function smoothing_filter(N, filter_width)
    @assert N >= filter_width && isodd(filter_width)
    filter = zeros(Float32, N, N)
    half_width = Int((filter_width - 1) / 2)
    for i in 1:half_width
        filter[i, 1:half_width+i] .= 1 / (half_width+i)
        filter[end+1-i, end-(half_width+i-1):end] .= 1 / (half_width+i)
    end

    for i in half_width+1:N-half_width
        filter[i, i-half_width:i+half_width] .= 1 / filter_width
    end

    return filter
end