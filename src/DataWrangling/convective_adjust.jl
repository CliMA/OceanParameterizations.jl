"""
    convective_adjust!(x)

Remove negative gradients from `x` (generally a temperature profile).
"""
function convective_adjust!(x)
    for i in length(x)-3:-1:2
        if x[i] > x[i+1]
            if x[i-1] > x[i]
                x[i] = x[i+1]
            else
                x[i] = (x[i-1] + x[i+1]) / 2
            end
        end
    end
end
