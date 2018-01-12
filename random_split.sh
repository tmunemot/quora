#/bin/bash

DEFAULT_LINE_COUNT=1000

_usage(){
cat << EOF
RANDOM_SPLIT

$(tput bold)NAME$(tput sgr0)
    $(tput bold)${0##*/}$(tput sgr0) -- randomly split a csv file into pieces

$(tput bold)SYNOPSIS$(tput sgr0)
    $(tput bold)${0##*/}$(tput sgr0) [$(tput bold)-l$(tput sgr0) $(tput smul)line_count$(tput rmul)] [$(tput smul)source$(tput rmul) [$(tput smul)name$(tput rmul)]] [$(tput smul)remainder$(tput rmul) [$(tput smul)name$(tput rmul)]] [$(tput smul)file$(tput rmul) $(tput smul)...$(tput rmul)]

$(tput bold)DESCRIPTION$(tput sgr0)
    The $(tput bold)${0##*/}$(tput sgr0) utility reads the given csv file and breaks it up into smaller files by randomly sampling.

    The options are as follows:

    $(tput bold)-l$(tput sgr0) line_count
        Create smaller files n lines in length (default: $DEFAULT_LINE_COUNT).

$(tput bold)EXAMPLES$(tput sgr0)
    The following is how to randomly split a source csv file into two sample files with one thousand lines and remainders
        ${0##*/} -l 1000 source.csv remainder.csv sample1.csv sample2.csv

EOF
}

line_count=$DEFAULT_LINE_COUNT

# parse inputs
while getopts ":hl:" OPT; do
    case $OPT in
        l)
            line_count=$OPTARG
            ;;
        h)
            _usage
            exit 1
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

src=${@:$OPTIND:1}
remainder=${@:$OPTIND+1:1}
samples=(${@:$OPTIND+2:$#})

# validate line counts
total=$(($(wc -l $src | awk '{print $1}')-1))
all_lines=$((${line_count} * ${#samples[@]}))
if [ "$all_lines" -ge "$total" ]; then
    echo "line_count too large or too many output files (total: $total lines, line_count: $line_count, output: ${#samples[@]} files)" >&2
    exit 1
fi

header=$(head -1 $src)

# copy contents of input file to a temporary file
tempfile=$(mktemp)
trap "rm -f $tempfile" EXIT
if [[ $(uname) == 'Linux' ]]; then
    # linux
    tail -n +2 $src | shuf > ${tempfile}
elif [[ $(uname) == 'Darwin' ]]; then
    # osx
    if [[ -z $(command -v gshuf) ]]; then
        echo "Missing gshuf. Install homebrew and run 'brew install coreutils'" >&2
        exit 1
    fi
    tail -n +2 $src | gshuf > ${tempfile}
else
    echo "Currently linux and osx are supported" >&2
    exit 1
fi

# create files contain randomly sampled lines
i=1
((j=${line_count}))
for sample in ${samples[@]}; do
    cat <(echo $header) <(sed -n "${i},${j}p" $tempfile) > $sample
    ((i+=${line_count}))
    ((j+=${line_count}))
done

# save all remainder
((i-=1))
cat <(echo $header) <(sed "1,${i}d" $tempfile) > $remainder
