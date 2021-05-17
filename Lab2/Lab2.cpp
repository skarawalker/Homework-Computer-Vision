#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    //String path = "../data/checkerboard_images/";
    String path;
    cout << "Pelase insert the path of the checkerboard images" << endl;
    cin >> path;
    if (path.empty()) {
        cout << "Unable to find images" << endl;
        return -1;
    }
    glob(path, filenames);

    vector<vector<Point3f>> points3d; //3D coordinates of the corners
    vector<vector<Point2f>> points2d; //2D cooridnates of the corners

    Size chessSize (6, 5);

    //Compute 3D coordinates of the corners.
    vector<Point3f> points;
    for (int i = 0; i < chessSize.height; i++)
    {
        for (int j = 0; j < chessSize.width; j++)
        {
            points.push_back(Point3f(0.11*j, 0.11*i, 0));
        }
    }

    cout << "Loading images..." << endl;

    Mat img;
    vector<Point2f> corners;
    bool finded;
    //Find corners
    for (const auto& fn : filenames)
    {
        img = imread(fn, IMREAD_GRAYSCALE); 
        finded = findChessboardCorners(img, chessSize, corners);

        if (finded)
        {
            points2d.push_back(corners);
            points3d.push_back(points);
        }
    }

    //drawChessboardCorners(img, chessSize, corners, finded);
    //imshow("Image", img);

    cout << "Calibrating, this might take a while..." << endl;

    Size imgSize = img.size();
    Mat cameraMatrix, distCoeffs, rvecs, tvecs;
    double calibration = calibrateCamera(points3d, points2d, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    //Print to output the calibration camera parameters
    cout << "Camera error : " << calibration << endl;
    cout << "Camera matrix : " << cameraMatrix << endl;
    cout << "Distortion coefficients : " << distCoeffs << endl;

    //Computes the mean reprojection error
    vector<Point2f> reprojected_points;
    vector<double> errors;
    for(int i = 0; i < filenames.size(); i++)
    {
        projectPoints(points, rvecs.row(i), tvecs.row(i), cameraMatrix, distCoeffs, reprojected_points);
        double sum_errors = 0;
        for (int j = 0; j < reprojected_points.size(); j++) {            
            sum_errors += norm(points2d[i][j] - reprojected_points[j]);
        }
        double mean_error = sum_errors / reprojected_points.size();
        errors.push_back(mean_error); 
    }
    
    cout << "Mean reprojection error: " << sum(errors)[0] / filenames.size() << endl;
    
    //Find the indexes of the images for which the calibration performs best and worst
    int bestIndex = 0;
    int worstIndex = 0;
    for (int i = 0; i < filenames.size(); i++) {
        if (errors[i] < errors[bestIndex])
            bestIndex = i;
        if (errors[i] > errors[worstIndex])
            worstIndex = i;
    }

    cout << "Best calibration image name: " << filenames[bestIndex] << ", with error: " << errors[bestIndex] << endl;
    cout << "Worst calibration image name: " << filenames[worstIndex] << ", with error: " << errors[worstIndex] << endl;

    //Visualize best and worst images
    Mat bestImg = imread(filenames[bestIndex]);
    Mat worstImg = imread(filenames[worstIndex]);
    resize(bestImg, bestImg, Size(bestImg.cols / 3, bestImg.rows / 3));
    resize(worstImg, worstImg, Size(worstImg.cols / 3, worstImg.rows / 3));
    imshow("Best image", bestImg);
    imshow("Worst image", worstImg);

    //Undistort the test image
    Mat testImage = imread("../data/test_image.png");
    Mat outputImage, R, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, testImage.size(), CV_32FC1, map1, map2);
    remap(testImage, outputImage, map1, map2, INTER_LINEAR);

    resize(testImage, testImage, Size(testImage.cols / 3, testImage.rows / 3));
    resize(outputImage, outputImage, Size(outputImage.cols / 3, outputImage.rows / 3));

    imshow("Original test image", testImage);
    imshow("Undistorted test image", outputImage);

    waitKey(0);
    return 0;
}