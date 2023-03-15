def numpy_to_video(numpy_array, output_file, fps=10, is_color=True):
    height, width = numpy_array.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    color_flag = cv2.COLOR_GRAY2BGR if not is_color else -1
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), is_color)

    for frame in numpy_array:
        if not is_color:
            frame = cv2.cvtColor(frame, color_flag)
        out.write(frame)

    out.release()