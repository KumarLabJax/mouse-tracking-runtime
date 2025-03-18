ffmpeg -f lavfi -i color=size=480x480:duration=60:rate=30:color=black -vf "drawtext=fontfile=/usr/share/fonts/truetype/cmu/cmunorm.ttf:fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h-text_h)/2:text='No Corners',drawtext=fontfile=/usr/share/fonts/truetype/cmu/cmunorm.ttf:fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h+text_h)/2:text='to Correct'" DEFAULT_VIDEO.mp4

