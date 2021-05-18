// Hough transform and Edge detection

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Parameters found with trackbars
#define LOW_THRESHOLD 350
#define HIGH_THRESHOLD 850

#define RHO 1
#define THETA CV_PI / 180
#define HOUGH_THRESHOLD 130

#define HIGHER_THRESHOLD 100
#define CENTER_THRESHOLD 25
#define MIN_DIST 1
#define MIN_RADIUS 0
#define MAX_RADIUS 10

Mat img;
vector<Vec2f> lines;
vector<Vec3f> circles;

void drawLines();
void drawCircle();

int main()
{
    Mat grayImg, cannyOutput;
    img = imread("../input.png");
    if (img.empty()) {
        cout << "Unable to find image" << endl;
        return -1;
    }
    imshow("Original image", img);

    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    //Canny edge detection
    Canny(grayImg, cannyOutput, LOW_THRESHOLD, HIGH_THRESHOLD);
    imshow("Canny Output", cannyOutput);

    //Detect lines
    HoughLines(cannyOutput, lines, RHO, THETA, HOUGH_THRESHOLD);

    //Detect circles  
    medianBlur(grayImg, grayImg, 3);
    HoughCircles(grayImg, circles, HOUGH_GRADIENT, 1, MIN_DIST, HIGHER_THRESHOLD, CENTER_THRESHOLD, MIN_RADIUS, MAX_RADIUS);

    //Draw lines
    drawLines();

    //Draw circle
    drawCircle();

    //Output image
    imshow("Output image", img);

    waitKey(0);
    return 0;
}

void drawLines()
{
    vector<Point> points, point;
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        points.push_back(pt1);
        points.push_back(pt2);
       
        //Uncomment to show the lines
        //line(img, pt1, pt2, Scalar(0, 0, 255), 1, 16);
        //imshow("Lines", img);
    }

    //calculate m1, m2, q1, q2 of the two lines
    float m1, m2, q1, q2;
    int xc, yc, yf, xf1, xf2;
    m1 = ((float)points[0].y - (float)points[1].y) / ((float)points[0].x - (float)points[1].x);
    m2 = ((float)points[2].y - (float)points[3].y) / ((float)points[2].x - (float)points[3].x);
    q1 = ((float)points[0].y - ((float)points[0].x * m1));
    q2 = ((float)points[2].y - ((float)points[2].x * m2));
  
    //calculating intersection point between two lines
    xc = ((q1 - q2) / (m2 - m1));
    yc = ((m1 * xc) + q1);
    //intersection with max y value for this image (yf = 374)
    yf = img.size().height - 1;
    xf1 = ((yf - q1) / m1);
    xf2 = ((yf - q2) / m2);

    point.push_back(Point(xc, yc));  //point1
    point.push_back(Point(xf1, yf));  //point2
    point.push_back(Point(xf2, yf));  //point3
    fillConvexPoly(img, point, Scalar(0, 0, 255), 8, 0);
}

void drawCircle()
{
    for (int i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img, center, radius, Scalar(0, 255, 0), -1, 8, 0);
    }
}