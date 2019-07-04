/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com

*/

#include "VehicleDetectionLib.h"
#include<opencv2/imgproc/imgproc.hpp>


int main(void)
{
	//string video_path("F:\\road_obstacle project\\recording.avi");
	string video_path("./road.avi");
	string train_path("./train/vehicle.xml");
	VehicleVision::params lparams;
	lparams.BIN_THRESH=130;		
	lparams.MIN_EDGE_TRESHOLD=50;		
	lparams.MAX_EDGE_TRESHOLD=200;		
	lparams.LINE_THRESH=3;			
	lparams.VEHICLE_LENGTH=4;			
	lparams.MAX_VEHICLE_HIST=20;		
	lparams.MIN_VEHICLES=15;			
	lparams.MAX_VEHICLE_FRAMES=15;		

	
	VehicleVision::VehicleDetector vd(video_path,train_path);
	vd.initParams(lparams);
	vd.detect();
	return 0;
}

