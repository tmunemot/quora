#/bin/bash

DEFAULT_LINE_COUNT=1000

_usage(){
cat << EOF
RANDOM_SPLIT

$(tput bold)NAME$(tput sgr0)
    $(tput bold)${0##*/}$(tput sgr0) -- randomly sample subsets from train.csv.zip

$(tput bold)SYNOPSIS$(tput sgr0)
    $(tput bold)${0##*/}$(tput sgr0) [$(tput bold)-l$(tput sgr0) $(tput smul)line_count$(tput rmul)] [$(tput smul)remainder$(tput rmul) [$(tput smul)name$(tput rmul)]] [$(tput smul)file$(tput rmul) $(tput smul)...$(tput rmul)]

$(tput bold)DESCRIPTION$(tput sgr0)
    $(tput bold)${0##*/}$(tput sgr0) samples subsets from train.csv.zip and save remainder and the subsets as csv files

    $(tput bold)-l$(tput sgr0) line_count
	lines per file (default: $DEFAULT_LINE_COUNT).

$(tput bold)EXAMPLES$(tput sgr0)
        ${0##*/} -l 1000 remainder.csv sample1.csv sample2.csv

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

remainder=${@:$OPTIND:1}
samples=(${@:$OPTIND+1:$#})
rsc=$(pipenv --where)/resource

if [ ! -f "$rsc/train.csv.zip" ]; then
    echo failed to load train.csv.zip 
    exit 1
else
    unzip $rsc/train.csv.zip -d $rsc

    # copy contents to a temporary file
    header=$(head -1 $rsc/train.csv)
    tempfile=$(mktemp)
    trap "rm -f $tempfile" EXIT
    if [[ $(uname) == 'Linux' ]]; then
        tail -n +2 $rsc/train.csv | shuf > ${tempfile}
	rm $rsc/train.csv
    elif [[ $(uname) == 'Darwin' ]]; then
        if [[ -z $(command -v gshuf) ]]; then
            echo "missing gshuf. install coreutils" >&2
            exit 1
        fi
        tail -n +2 $rsc/train.csv | gshuf > ${tempfile}
	rm $rsc/train.csv
    else
        echo "only linux and macOS are supported" >&2
        exit 1
    fi
fi

# validate line counts
total=$(($(wc -l ${tempfile} | awk '{print $1}')-1))
all_lines=$((${line_count} * ${#samples[@]}))
if [ "$all_lines" -ge "$total" ]; then
    echo "line_count exceeds a limit or too many outputs are specified (total: $total lines, line_count: $line_count, output: ${#samples[@]} files)" >&2
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
