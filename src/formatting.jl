const RED = "\x1b[31m"
const GREEN = "\x1b[32m"
const YELLOW = "\x1b[33m"
const BLUE = "\x1b[34m"
const RESET = "\x1b[0m"

export RED, GREEN, YELLOW, BLUE, RESET, visualize_comparison

using PrettyTables

const COLORS = [
    :red           ,
    :green         ,
    :yellow        ,
    :blue          ,
    :magenta       ,
    :cyan          ,
    :light_red     ,
    :light_green   ,
    :light_yellow  ,
    :light_blue    ,
    :light_magenta ,
    :light_cyan    ,
    :black         ,
]

function visualize_comparison(dataframe::DataFrame)
    hs = ()
    c = 0
    for value in dataframe.y
        hs = (Highlighter((data, i, j) -> data[i, j] == value, bold=true, foreground=COLORS[c % length(COLORS) + 1]), hs...)
        c += 1
    end
    pretty_table(dataframe, highlighters=(hs))
end