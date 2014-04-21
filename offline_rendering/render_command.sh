# for fast render, ssh into several machines to parallelize frames
#./mitsuba -xj 2 ~/offline_renders/cbox/cbox_*.xml
# bash render_command.sh
# this currently has some issues.

source /contrib/projects/Mitsuba-Renderer/mitsuba/setpath.sh &&
mitsuba -c evjang@ssh.cs.brown.edu:/contrib/projects/Mitsuba-Renderer/mitsuba -xj 2 ~/offline_renders/cbox/cbox_teapot_0024.xml

# avconv -f image2 -i img%d.jpg /tmp/a.mpg
