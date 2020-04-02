## Things I export to my bashrc
alias c="clear"

#Yeeepppp
PS1='\[\033[1;34m\]\W →  \[\033[m\]'

#Tmux is nothing without colours
[[ $TERM != "screen-256color" ]] && exec tmux -u

#Yep
eval "$(thefuck --alias)"
