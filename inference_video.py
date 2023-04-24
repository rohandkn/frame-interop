import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from rife.model.pytorch_msssim import ssim_matlab
import sys
sys.path.append("/home/mverghese/Documents/vignesh-VLR/sketch2vid/rife")

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

def run_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    args['fp16'] = False
    args['modelDir'] = 'train_log'
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print(args)
        if(args['fp16']):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        try:
            try:
                from rife.model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(args['modelDir'], -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(args['modelDir'], -1)
                print("Loaded v3.x HD model.")
        except:
            from rife.model.RIFE_HD import Model
            model = Model()
            model.load_model(args['modelDir'], -1)
            print("Loaded v1.x HD model")
    except:
        from rife.model.RIFE import Model
        model = Model()
        model.load_model(args['modelDir'], -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()

    if not args['video'] is None:
        videoCapture = cv2.VideoCapture(args['video'])
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(tot_frame)
        videoCapture.release()
        if args['fps'] is None:
            fpsNotAssigned = True
            args['fps'] = fps * (2 ** args['exp'])
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(args['video'])
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(args['video'])
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key= lambda x:int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    if args.png:
        if not os.path.exists('vid_out'):
            os.mkdir('vid_out')
    else:
        if args.output is not None:
            vid_out_name = args.output
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)
