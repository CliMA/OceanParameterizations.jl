using ArgParse
using WindMixing

s = ArgParseSettings()
@add_arg_table s begin
    "--input"
        help = "location of input file in training_output directory"
        required = true
    "--output"
        help = "destination of output file in extracted_training_output directory"
        required = true
    "--type"
        help = "training type: NDE or NN"
        default = "NDE"
end

arg_parse = parse_args(s)

FILE_PATH = joinpath(pwd(), "training_output", arg_parse["input"])
OUTPUT_PATH = joinpath(pwd(), "extracted_training_output", arg_parse["output"])
OUTPUT_PATH = joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output", arg_parse["output"])

type = arg_parse["type"]

extract_NN(FILE_PATH, OUTPUT_PATH, type)

