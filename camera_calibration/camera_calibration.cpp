#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

static const int board_width = 9;
static const int board_height = 14;
static int square_size = 100;

static const std::string input_image_list = "/Users/sofiawu/Work/2017/camera_calibration/camera_calibration/VID5.xml";
static const std::string output_file = "/Users/sofiawu/Work/2017/camera_calibration/camera_calibration/out.xml";

static bool ReadStringList( const string& filename, vector<string>& l ) {
    l.clear();
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

static void CalcBoardCornerPositions(Size board_size, float squareSize,
                                     vector<Point3f>& corners){
    corners.clear();
    
    for( int i = 0; i < board_size.height; ++i )
        for( int j = 0; j < board_size.width; ++j )
            corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
}

static double ComputeReprojectionErrors( const vector<vector<Point3f> >& object_points,
                                        const vector<vector<Point2f> >& image_points,
                                        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                        const Mat& camera_matrix , const Mat& dist_coeffs,
                                        vector<float>& per_view_errors) {
    std::vector<cv::Point2f> image_points_temp;
    size_t total_points = 0;
    double total_err = 0, err;
    per_view_errors.resize(object_points.size());
    
    for(size_t i = 0; i < object_points.size(); ++i ) {
        cv::projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, image_points_temp);
        err = cv::norm(image_points[i], image_points_temp, NORM_L2);
        
        size_t n = object_points[i].size();
        per_view_errors[i] = (float) std::sqrt(err * err / n);
        total_err        += err * err;
        total_points     += n;
    }
    
    return std::sqrt(total_err / total_points);
}


// Print camera parameters to the output file
static void SaveCameraParams( Size& image_size, Mat& camera_matrix, Mat& dist_coeffs,
                             const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                             const vector<float>& reproj_errs,
                             const vector<vector<Point2f> >& image_points,
                             double total_avg_err ) {
    FileStorage fs( output_file, FileStorage::WRITE );
    
    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf), "%c", t2 );
    
    fs << "calibration_time" << buf;
    
    if( !rvecs.empty() || !reproj_errs.empty() )
        fs << "nr_of_frames" << (int)std::max(rvecs.size(), reproj_errs.size());
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    
    fs << "board_width" << board_width;
    fs << "board_height" << board_height;
    fs << "square_size" << square_size;
    
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    
    fs << "avg_reprojection_error" << total_avg_err;
    if (!reproj_errs.empty())
        fs << "per_view_reprojection_errors" << Mat(reproj_errs);
}

static bool RunCalibration(cv::Size image_size, cv::Mat& camera_matrix, cv::Mat& dist_coeffs,
                           std::vector<std::vector<cv::Point2f> > image_points) {
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reproj_errs;
    
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    
    std::vector<std::vector<cv::Point3f> > object_points(1);
    CalcBoardCornerPositions(cv::Size(board_width, board_height), square_size, object_points[0]);
    object_points.resize(image_points.size(), object_points[0]);
    
    double rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, 0);
    std::cout << "Re-projection error reported by calibrateCamera: "<< rms << std::endl;
    
    bool ok = cv::checkRange(camera_matrix) && cv::checkRange(dist_coeffs);
    
    double total_avg_err = ComputeReprojectionErrors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs, reproj_errs);
    
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
    << ". avg re projection error = " << total_avg_err << endl;
    
    if (ok)
        SaveCameraParams(image_size, camera_matrix, dist_coeffs, rvecs, tvecs, reproj_errs, image_points, total_avg_err);
    return ok;
}

int main(int argc, char* argv[]) {
    vector<vector<Point2f> > images_all_points;
    Mat camera_matrix, dist_coeffs;
    Size image_size;
    vector<std::string> image_list;
    
    ReadStringList(input_image_list, image_list);
    
    //detect corners
    int chessboard_flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;
    for(int i = 0; i < image_list.size(); ++i) {
        cv::Mat img = cv::imread(image_list[i], 0);
        image_size = img.size();
        
        std::vector<cv::Point2f> image_points;
        bool found = cv::findChessboardCorners( img, cv::Size(board_width, board_height), image_points, chessboard_flags);
        if(found) {
            cv::cornerSubPix( img, image_points, cv::Size(11, 11),
                             cv::Size(-1, -1), cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1 ));
            //cv::find4QuadCornerSubpix(img, image_points, Size(11,11));
            images_all_points.push_back(image_points);
        }
        //show
        //cv::drawChessboardCorners( img, cv::Size(board_width, board_height), cv::Mat(image_points), found );
        //imshow("Image View", img);
        //waitKey(500);
    }

    RunCalibration(image_size, camera_matrix, dist_coeffs, images_all_points);

    return 0;
}
