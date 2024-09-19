import cv2

def print_supported_codecs():
    fourcc_list = [
        'avc1', 'mp4v', 'xvid', 'mjpg', 'h264',
        'hevc', 'vp80', 'vp90', 'mp4v', 'div3',
        'divx', 'u263', 'i263', 'x264'
    ]

    supported_codecs = []
    for fourcc_str in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter('test.mp4', fourcc, 30, (640, 480))
        
        if out.isOpened():
            supported_codecs.append(fourcc_str)
            print(f"Codec {fourcc_str} is supported")
            out.release()
            print("supported_codecs:", supported_codecs)
        else:
            print(f"Codec {fourcc_str} is not supported")

    ## 所有支持的编码器包括：
    ## avc1, mp4v, xvid, mjpg, h264, hevc, vp80, vp90, mp4v, div3, divx, u263, i263, x264
    print("所有支持的编码器包括：")
    print(supported_codecs)
print_supported_codecs()