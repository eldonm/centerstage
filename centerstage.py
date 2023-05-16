import dlib
import cv2
import os
import moviepy.editor as mp
import tempfile
from natsort import natsorted
import argparse

#process command line args
def process_arguments():
    parser = argparse.ArgumentParser(description="Crops in and centers a subject's face given an input video")

    # Add command-line arguments
    parser.add_argument('-f', '--file', help='Path to the input video file')
    parser.add_argument('-o', '--output', help='Destination folder for output video file')

    # Parse the arguments
    args = parser.parse_args()

    # Access the values of the arguments
    video_path = args.file
    output_path = args.output

    if not output_path:
        #use current working directory for output by default
        output_path = os.getcwd()

    if not video_path:
        print(f"Missing video path. Type -h for help.")
        exit
    elif not os.path.exists(video_path):
        print(f"Invalid file path.")
        exit
    else:
        filename = os.path.basename(video_path)
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            #process arguments
            process_center_stage_video(video_path, output_path)
        else:
            print(f"Invalid file type. MP4 or AVI files only.")


#driver function to prepare center stage video
def process_center_stage_video(video_path, output_path):
    
    if os.path.exists(video_path):
        filename = os.path.basename(video_path)
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            #create working temp directory
            temp_dir = create_temp_directory()

            #extract keyframes
            keyframe_path = os.path.join(temp_dir, "keyframes")
            # create the directory
            os.makedirs(keyframe_path, exist_ok=True)
            fps = extract_keyframes(video_path, keyframe_path)

            #align and crop keyframes
            aligned_keyframe_path = os.path.join(temp_dir, "aligned_keyframes")
            # create the directory
            os.makedirs(aligned_keyframe_path, exist_ok=True)
            align_and_crop_faces(keyframe_path, aligned_keyframe_path)

            #extract audio
            audio_path = os.path.join(temp_dir, f"audio.aac")
            extract_audio(video_path, audio_path)

            #produce the video from keyframes
            compose_video_from_keyframes(aligned_keyframe_path, f"{temp_dir}/video.mp4", fps)
            
            #merge audio and produce final video
            add_audio_to_video(f"{temp_dir}/video.mp4", audio_path, f"{output_path}/{filename}", fps)

#process an entire directory of videos
def process_center_stage_video_directory(video_directory, output_path):

    # Iterate over the keyframes in the directory
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(video_directory, filename)
            process_center_stage_video(video_path, output_path)
            

#extracts keyframes from video
def extract_keyframes(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        return
    
    keyframe_count = 0

    # Get the framerate of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # Check if frame reading was successful
        if not ret:
            break

        # Save the keyframe to an image file
        keyframe_filename = f"{output_path}/keyframe_{keyframe_count}.jpg"
        cv2.imwrite(keyframe_filename, frame)
        keyframe_count += 1

    # Release the video capture object
    video.release()
    print(f"Keyframes extracted: {keyframe_count} at {fps} fps")

    return fps


#extracts audio from video
def extract_audio(video_path, output_path):
    # Load the video file
    video = mp.VideoFileClip(video_path)

    # Extract the audio from the video
    audio = video.audio

    if audio:
        # Write the extracted audio to the output file
        audio.write_audiofile(output_path, codec="aac")


#produces aligned, cropped keyframes from original keyframes
def align_and_crop_faces(keyframes_directory, output_directory):
    # Load the face detection and alignment models from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Iterate over the keyframes in the directory
    for filename in os.listdir(keyframes_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            keyframe_path = os.path.join(keyframes_directory, filename)
            
            # Load the image using Dlib
            img = dlib.load_rgb_image(keyframe_path)

            # Detect faces in the keyframe
            dets = detector(img)

            num_faces = len(dets)
            if num_faces != 0:
                # Find the 5 face landmarks we need to do the alignment.
                faces = dlib.full_object_detections()
                for detection in dets:
                    faces.append(predictor(img, detection))

                # Get the aligned face images
                images = dlib.get_face_chips(img, faces, size=512, padding=0.75)
                
                for image in images:
                    # Save the aligned and cropped face region to the output directory
                    output_filename = os.path.join(output_directory, f"aligned_{filename}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(output_filename, image)


#merges keyframes with original audio
def compose_video_from_keyframes(keyframes_directory, output_path, fps):

    frameSize = (512, 512)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

    # Iterate over the keyframes in the directory
    sorted_files = natsorted(os.listdir(keyframes_directory))
    for filename in sorted_files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            keyframe_path = os.path.join(keyframes_directory, filename)
            img = cv2.imread(keyframe_path)
            out.write(img)
           
    out.release()


def add_audio_to_video(video_path, audio_path, output_path, fps):
    video = mp.VideoFileClip(video_path)
    video = video.set_fps(fps)

    if os.path.exists(audio_path):
        audio = mp.AudioFileClip(audio_path)
        video = video.set_audio(audio)
    
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")


#creates a temporary working directory
def create_temp_directory():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    return temp_dir


# Call the function to process the arguments
process_arguments()