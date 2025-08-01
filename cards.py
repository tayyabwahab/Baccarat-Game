
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors
from utils.torch_utils import select_device, load_classifier, time_sync
from CentroidTrackerFinal import CentroidTracker


@torch.no_grad()
def plot_one_box(objectID,value, im, color=(0, 0, 255), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    #assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    centroid=value[0]
    x=value[1]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    label=value[2]
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    text = "ID {}".format(objectID)
    cv2.putText(im, text, (centroid[0] - 10, centroid[1] - 10),             
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.circle(im, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def run():
    weights='model.pt'  # model.pt path(s)
    #source='IMG_9999.MOV'
    #save_path_vid='/mnt/sda2/card_detection/Cards-Project-github/output/'
    source='0'  # put path for video and  0 for webcam
    imgsz = 640  # Image size (640 or 480 or 384)
    conf_thres=0.60  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=10  # maximum detections per image
    device=''  # cuda device, i.e. '' or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=True  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project='runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    counting = 0
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    print('Libraries Imported')
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        print('Loading Model...')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        print('Model Loaded')
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        print('Starting live Streaming')
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time.time()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time.time()
        print(f'Detection:  ({t2 - t1:.3f}s)')

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwhqqqqqqqqqqqq
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                    
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                       # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
        #if len(det):
            rects = det[:,0:4]
            if counting==0:
                 orig_size=im0.shape
                 ct = CentroidTracker(orig_size,7)
                 #w,h= orig_size[1],orig_size[0]
                 #vid_writer = cv2.VideoWriter(save_path_vid+'videotagged.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
# elif counting >= 3:
            t3 = time.time()
            objects = ct.update(rects)
            t4 = time.time()
#print(f'Tracking:  ({t6 - t5:.3f}s)')
            for (objectID, value) in objects.items():
                  plot_one_box(objectID,value, im0, line_thickness=line_thickness)
            
            cnt = []
            lab = []
            ids = []
            #print('value = ',len(value))
            for (objectID, value) in objects.items():
              #if len(value) < 3:
              #  break
                plot_one_box(objectID,value, frame, line_thickness=line_thickness)
                cnt.append(value[0])
                lab.append(value[2])
                ids.append(objectID)
            print('Frame ',counting-1)
            for i in range(len(cnt)):
              print(str(ids[i])+' '+lab[i]+' location = '+str(cnt[i]))
            
            counting=counting+1
            #vid_writer.write(im0)
            if webcam==False:
                cv2.imshow('frame',im0)
                cv2.waitKey(1)
            
              

            # Print time (inference + NMS)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        if view_img:
            t5 = time.time()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
            t6 = time.time()
            print(f'Image View:  ({t6 - t5:.3f}s)')
    
        # Save results (image with detections)
        
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
        t7 = time.time()
        
        print(f'Total Time:  ({t7 - t1:.3f}s)')
        print('****************************************************')
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model ( to fix SourceChangeWarning)

    #print(f'Done. ({time.time() - t0:.3f}s)')
    

if __name__ == "__main__":
    ts = time.time()
    run()
    te = time.time()
    cv2.destroyAllWindows()
    s=''
    print(f'Total Streaming Time: {s}  ({te - ts:.3f}s)')
