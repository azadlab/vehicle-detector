/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com

*/

#include "VehicleDetectionLib.h"
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv/cv.h>

#include <math.h>
#include "utils.h"

namespace VehicleVision
{
	struct Vehicle {
	CvPoint vmin, vmax;
	int horiz_sym;
	bool isvalid;
	unsigned int prevUpdate;
	Mat patch;
};

struct CandidateVehicle {
	CvPoint vehicle_center;
	float vehicle_radius;
	unsigned int imgIdx;
	int vehicleIdx;
	Mat patch;
};
std::vector<Vehicle> vehicles;
std::vector<CandidateVehicle> candidates;
int areEdgesSymmetric(IplImage* img, CvPoint vmin, CvPoint vmax) {
  
  float val = 0;
  int horizAxis = -1; 
  
  int xmin = vmin.x;
  int ymin = vmin.y;
  int xmax = vmax.x;
  int ymax = vmax.y;
  int width = img->width/2;
  int maxIdx = 1;

  int x=xmin;
  for(int k=0; k<xmax; k++) {
	float symVal = 0;
    for(int y=ymin; y<ymax; y++) {
		int rowIdx = y*img->width*img->nChannels;
        for(int s=1; s<width; s++) {
          int minus = x-s;
          int positive = x+s;
		  unsigned char Vminus = (minus < xmin) ? 0 : (unsigned char)img->imageData[rowIdx+minus*img->nChannels];
          unsigned char Vpos = (positive >= xmax) ? 0 : (unsigned char)img->imageData[rowIdx+positive*img->nChannels];
          symVal += abs(Vminus-Vpos);
        }
    }

	if (horizAxis == -1 || val > symVal) { 
		horizAxis = x;
		val = symVal;
	}
	x++;
  }

  return horizAxis;
}

unsigned char getPixel(IplImage* img, int x, int y) {
	return (unsigned char)img->imageData[(y*img->width+x)*img->nChannels];
}

bool hasVertResponse(IplImage* edges, int x, int y, int vert_min, int vert_max,int BW_THRESH) {
	bool binPix = (getPixel(edges, x, y) > BW_THRESH);
	if (y-1 >= vert_min) binPix &= (getPixel(edges, x, y-1) < BW_THRESH);
	if (y+1 < vert_max) binPix &= (getPixel(edges, x, y+1) < BW_THRESH);
	return binPix;
}

int isHorizontal(IplImage* edges, int x, int y, CvPoint vmin, CvPoint vmax, int maxHorzDist,int BW_THRESH) {

	int rightIdx = 0;
	int dist = maxHorzDist;
	for (int xx=x; xx<vmax.x; xx++) {
		if (hasVertResponse(edges, xx, y, vmin.y, vmax.y,BW_THRESH)) {
			rightIdx++;
			dist = maxHorzDist; 
		} else {
			dist--;
			if (dist <= 0) {
				break;
			}
		}
	}

	int leftIdx = 0;
	dist = maxHorzDist;
	for (int xx=x-1; xx>=vmin.x; xx--) {
		if (hasVertResponse(edges, xx, y, vmin.y, vmax.y,BW_THRESH)) {
			leftIdx++;
			dist = maxHorzDist; 
		} else {
			dist--;
			if (dist <= 0) {
				break;
			}
		}
	}

	int res = leftIdx+rightIdx;
	return res;
}


bool isValid(IplImage* img, IplImage* edges, Vehicle* veh, int& idx,params lparams) {

	idx = -1;

	veh->horiz_sym = areEdgesSymmetric(img, veh->vmin, veh->vmax);
	if (veh->horiz_sym == -1) return false;

	int horizLines = 0;
	for (int i = veh->vmin.y; i < veh->vmax.y; i++) {		
		if (isHorizontal(edges, veh->horiz_sym, i, veh->vmin, veh->vmax, 2,lparams.BIN_THRESH) > lparams.VEHICLE_LENGTH) {
			horizLines++;
		}
	}

	int vert_mid = (veh->vmax.y + veh->vmin.y)/2;

	int numNeigh = 0;
	float bestDist = 0;

	for (int k = 0; k < candidates.size(); k++) {
		int dx = candidates[k].vehicle_center.x - veh->horiz_sym;
		int dy = candidates[k].vehicle_center.y - vert_mid;
		float dist = dx*dx + dy*dy;
		
		if (dist <= candidates[k].vehicle_radius*candidates[k].vehicle_radius) {
			numNeigh++;
			if (idx == -1 || dist < bestDist) {
				idx = candidates[k].vehicleIdx;
				bestDist = dist;
			}
		}
		
	}
	
	return ((horizLines >= lparams.LINE_THRESH || numNeigh >= lparams.MIN_VEHICLES) );
}

void discardOldVehicles(unsigned int curImg,int MAX_VEHICLES) {
	
	std::vector<CandidateVehicle> candidate;
	for (int i = 0; i < candidates.size(); i++) {
		if (curImg - candidates[i].imgIdx < MAX_VEHICLES) {
			candidate.push_back(candidates[i]);
		}
	}
	candidates = candidate;
}

void deleteCandidate(int candidate_idx) {

	std::vector<CandidateVehicle> cand;
	for (int i = 0; i < candidates.size(); i++) {
		if (candidates[i].vehicleIdx != candidate_idx) {
			cand.push_back(candidates[i]);
		}
	}
	candidates = cand;
}

void discardLostVehicles(unsigned int curImg,int VEHICLE_LOST) {

	for (int i=0; i<vehicles.size(); i++) {
		if (vehicles[i].isvalid && curImg - vehicles[i].prevUpdate >= VEHICLE_LOST) {
			deleteCandidate(i);
			vehicles[i].isvalid = false;
		}
	}
}

void findVehicles(IplImage* img, CvHaarClassifierCascade* haarcascade, CvMemStorage* haarMem,params lparams) {

	static unsigned int imgIdx = 0;
	imgIdx++;
	Mat tmp_img = cv::cvarrToMat(img,true);

	discardOldVehicles(imgIdx,lparams.MAX_VEHICLE_HIST);

	const double sfact = 1.25; 
	const int neigh = 2; 
	CvSeq* ROIs = cvHaarDetectObjects(img, haarcascade, haarMem, sfact, neigh, CV_HAAR_DO_CANNY_PRUNING,cvSize(img->width/10,img->height/10),cvSize(img->width/2,img->height/2));

	if (ROIs->total > 0) {
		
		IplImage *edges = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		cvCanny(img, edges, lparams.MIN_EDGE_TRESHOLD, lparams.MAX_EDGE_TRESHOLD);
		//cvShowImage("Canny Image",edges);

		for (int i = 0; i < ROIs->total; i++) {
			CvRect* ROI = (CvRect*)cvGetSeqElem(ROIs, i);
			
			Vehicle v;
			v.vmin = cvPoint(ROI->x, ROI->y);
			v.vmax = cvPoint(ROI->x + ROI->width, ROI->y + ROI->height);
			v.isvalid = true;
			v.patch = Mat(ROI->height,ROI->width,CV_8UC3);
			tmp_img(Rect(ROI->x,ROI->y,ROI->width,ROI->height)).copyTo(v.patch);  

			
			int idx;
			bool validVehicle = isValid(img, edges, &v, idx,lparams);
			//("Vehicle Valid:%d\n",validVehicle);
			if (validVehicle) { 
				if (idx == -1) { 

					v.prevUpdate = imgIdx;

					for(int k=0; k<vehicles.size(); k++) {
						if (vehicles[k].isvalid == false) {
							idx = k;
							break;
						}
					}
					if (idx == -1) { 
						idx = vehicles.size();
						vehicles.push_back(v);
					}
					
				} else {
					
					vehicles[idx] = v;
					vehicles[idx].prevUpdate = imgIdx;
					
				}

				CandidateVehicle veh_cand;
				veh_cand.imgIdx = imgIdx;
				veh_cand.vehicleIdx = idx;
				veh_cand.vehicle_radius = (MAX(ROI->width, ROI->height))/4; 
				veh_cand.vehicle_center = cvPoint((v.vmin.x+v.vmax.x)/2, (v.vmin.y+v.vmax.y)/2);
				veh_cand.patch = Mat(ROI->height,ROI->width,CV_8UC3);
				tmp_img(Rect(ROI->x,ROI->y,ROI->width,ROI->height)).copyTo(veh_cand.patch); 
				candidates.push_back(veh_cand);
			}
		}

		cvReleaseImage(&edges);
	} 

	discardLostVehicles(imgIdx,lparams.MAX_VEHICLE_FRAMES);

}
	void VehicleDetector::initParams()
	{
		
		lparams.BIN_THRESH=250;		
		lparams.MIN_EDGE_TRESHOLD=1;		
		lparams.MAX_EDGE_TRESHOLD=100;		
		lparams.LINE_THRESH=4;			
		lparams.VEHICLE_LENGTH=10;			
		lparams.MAX_VEHICLE_HIST=30;		
		lparams.MIN_VEHICLES=28;			
		lparams.MAX_VEHICLE_FRAMES=10;		


	}
	void VehicleDetector::initParams(params lparams)
	{
		this->lparams = lparams;

	}

	void drawVehicles(Mat frame,CvPoint scale,CvPoint trans) {

			CvPoint p1=cvPoint(0,0);
			CvPoint p2=cvPoint(0,0);
			
			double error=0;
			for (int i = 0; i < vehicles.size(); i++) {
				Vehicle* v = &vehicles[i];

				Mat half_frame = Mat(frame.cols/2,frame.rows/2,CV_8UC3);
				pyrDown(frame,half_frame,cvSize(frame.cols/2,frame.rows/2));
				
				int minWidth = MIN(v->patch.cols,v->vmax.x-v->vmin.x);
				int minHeight = MIN(v->patch.rows,v->vmax.y-v->vmin.y);
				int midPointX = (v->vmax.x+v->vmin.x)/2;
				int midPointY = (v->vmax.y+v->vmin.y)/2;
				Mat cpatch = Mat(minHeight,minWidth,CV_8UC1);
				half_frame(Rect((midPointX-minWidth/2)+trans.x,(midPointY-minHeight/2)+trans.y/2,minWidth,minHeight)).copyTo(cpatch); 
				midPointX = v->patch.rows/2;
				midPointY = v->patch.cols/2;
				Mat vpatch;
				v->patch(Rect((midPointX-minWidth/2),(midPointY-minHeight/2),minWidth,minHeight)).copyTo(vpatch); 
				
				Mat res;
				//matchTemplate(vpatch,cpatch,res,CV_TM_CCORR_NORMED);
				//ccor = sum(res)[0];
				error = (norm(vpatch,cpatch,CV_L2))/(double)(vpatch.rows*vpatch.cols);
				//imshow("vpatch",vpatch);imshow("cpatch",cpatch);
				//cv::waitKey(0);
				cout << "error:" << error << endl;
				if (v->isvalid && error<1.1) {
					p1.x = v->vmin.x*scale.x+trans.x;
					p1.y = v->vmin.y*scale.y+trans.y;
					p2.x = v->vmax.x*scale.x+trans.x;
					p2.y = v->vmax.y*scale.y+trans.y;
					rectangle(frame, p1, p2, Scalar(0,255,0), 2);
					
					//int midY = (v->vmin.y + v->vmax.y) / 2;
					
				}
			}
			
	}

	VehicleDetector::VehicleDetector(std::string video_path,string train_path,bool UseCamera,bool WriteOutput,string out_path)
	{
		
		initParams();
		if(UseCamera)
		{
			input = VideoCapture(0);
		}
		else
		{
			this->input_video_path = video_path;
			this->lparams = lparams;
			input = VideoCapture(input_video_path); 
		}

		if (!input.isOpened()) {
			fprintf(stderr, "Error: Can't open video\n");
			return;
		}
		this->train_path = train_path;
		cv::Size vid_size;
		vid_size.height = (int) input.get(CV_CAP_PROP_FRAME_HEIGHT);
		vid_size.width = (int) input.get(CV_CAP_PROP_FRAME_WIDTH);
		float fps = (int) input.get(CV_CAP_PROP_FPS);
		if(WriteOutput)
		{
			this->output_video_path = out_path;
			output = VideoWriter(output_video_path,-1,fps,vid_size);
			if(!output.isOpened())
			{
			fprintf(stderr, "Error: Can't write output video\n");
			return;
			}
		}
	}


	void VehicleDetector::detect()
	{
		
		CvMemStorage* haarMem = cvCreateMemStorage(0);
		CvHaarClassifierCascade* haarcascade = (CvHaarClassifierCascade*)cvLoad(train_path.c_str());
		Mat frame;
		int width = input.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = input.get(CV_CAP_PROP_FRAME_HEIGHT);
		Size frame_size = Size(width/2, height/2);
		Mat half_frame = Mat(frame_size,CV_8UC3);
		int offset = 50;
		Mat lhor_frame = Mat(frame_size.height/2,frame_size.width-2*offset,CV_8UC3);
		//Mat edges = Mat(frame_size, CV_8UC1);
		float fps = (int) input.get(CV_CAP_PROP_FPS);
		int key_pressed = 0;
		while(key_pressed != 27) {
			
		input>>frame;
		
		if(frame.empty())
			return;
		
		if(key_pressed==32)
			cvWaitKey(0);

		pyrDown(frame,half_frame,frame_size);
		half_frame(Rect(0,frame_size.height/2,frame_size.width-2*offset,frame_size.height/2)).copyTo(lhor_frame);
		IplImage* ipl_frame = cvCreateImage(cvSize(lhor_frame.cols,lhor_frame.rows),8,3);
IplImage * ipl_tmp = new IplImage(lhor_frame);
cvCopy(&ipl_tmp,ipl_frame);
//cvCloneImage(&(IplImage)lhor_frame);
		
		findVehicles(ipl_frame, haarcascade, haarMem,lparams);
		drawVehicles(frame,cvPoint(2,2),cvPoint(0,frame_size.height));
		imshow("VehicleDetector",frame);
		
		if(!output_video_path.empty())
			output.write(frame);
		float fps = (int) input.get(CV_CAP_PROP_FPS);
		key_pressed=waitKey(1000/fps);
		cvReleaseImage(&ipl_frame);
		}
	}
	
}
