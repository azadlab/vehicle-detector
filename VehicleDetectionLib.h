/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com

*/

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace VehicleVision
{

	typedef struct params
	{
	
		int BIN_THRESH;				// Threshold to declare a pixel white in edge image
		int MIN_EDGE_TRESHOLD;		// minimum value for edge detection threshold
		int MAX_EDGE_TRESHOLD;		// Max value for edge detection threshold
		int LINE_THRESH;			    // Minimum no of lines in a region to declare it a vehicle
		int VEHICLE_LENGTH;				  // Minimum horizontal Length of a vehicle
		int MAX_VEHICLE_HIST;		      // Maximum no of vehicle hypothesis to keep
		int MIN_VEHICLES;				// Minimum no of samples to be detected out of total detected hypothesis resulting in probability of positive samples.
		int MAX_VEHICLE_FRAMES;			// No of frames before a detected vehicle hypothesis is discarded 
		
	}params;

	class  VehicleDetector
	{
	private:
		std::string input_video_path;
		std::string output_video_path;
		string train_path;
		cv::VideoCapture input;
		cv::VideoWriter output;
		params lparams;
	public:
		
		VehicleDetector() {};
		void initParams();
		void initParams(params lparams);
		VehicleDetector(std::string video_path,string train_path,bool UseCamera=false,bool WriteOutput=false,string out_path="");
		void detect(void);
		
	};

}
