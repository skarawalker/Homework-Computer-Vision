#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//function to extract the keypoints
vector<KeyPoint> extractKeypoints(Mat src)
{
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints;
    detector->detect(src, keypoints);
    return keypoints;
}

//function to extract the descriptors
Mat extractDescriptors(Mat src, vector<KeyPoint> keypoints)
{
    Ptr<SIFT> detector = SIFT::create();
    Mat descriptor;
    detector->compute(src, keypoints, descriptor);
    return descriptor;
}

//function to match the descriptors in an image
vector<vector<DMatch>> findMatches(vector<Mat> objectsdescriptors, Mat frameDescriptors)
{
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    vector<vector<DMatch>> dmatches;
    for (Mat descriptors : objectsdescriptors)
    {
        vector<DMatch> matches;
        matcher->match(descriptors, frameDescriptors, matches);
        dmatches.push_back(matches);
    }
    return dmatches;
}

//function to find the homography matrices
vector<Mat> findPointsHomographies(vector<vector<KeyPoint>> objKeypoints, vector<KeyPoint> frameKeypoints, vector<vector<DMatch>> matches, vector<vector<uint8_t>>& masks)
{
    vector<Mat> homographies;
    for (int i = 0; i < matches.size(); ++i)
    {
        vector<Point2f> objectPoints, framePoints;
        vector<KeyPoint> objectKeypoint = objKeypoints[i];
        for (int j = 0; j < matches[i].size(); ++j)
        {
            objectPoints.push_back(objectKeypoint[matches[i][j].queryIdx].pt);
            framePoints.push_back(frameKeypoints[matches[i][j].trainIdx].pt);
        }
        vector<uint8_t> mask;
        Mat H = findHomography(objectPoints, framePoints, RANSAC, 3, mask);
        homographies.push_back(H);
        masks.push_back(mask);
    }
    return homographies;
}

//function to compute the corners in a frame given the input corners and the homographies
vector<vector<Point2f>> computeRectCorners(vector<vector<Point2f>> obj_corners, vector<Mat> homographies)
{
    vector<vector<Point2f>> scene_corners;
    for (int i = 0; i < obj_corners.size(); ++i)
    {
        vector<Point2f> corners;
        perspectiveTransform(obj_corners[i], corners, homographies[i]);
        scene_corners.push_back(corners);
    }
    return scene_corners;
}

//function to draw a rectangular boundary in the image
void drawRect(Mat image, vector<Point2f> corners, Scalar color = Scalar(0, 0, 255), int lineWidth = 4)
{
    line(image, corners[0], corners[1], color, lineWidth);
    line(image, corners[1], corners[2], color, lineWidth);
    line(image, corners[2], corners[3], color, lineWidth);
    line(image, corners[3], corners[0], color, lineWidth);
}

int main(int argc, char* argv[])
{
    //get from the command line the paths
    vector<Mat> frames;
    vector<Mat> objects;
    if (argc != 3) {
        cout << "USAGE: $" << argv[0] << " IMAGES_PATH VIDEO_PATH" << endl;
        cout << "IMAGES_PATH: path where the folder containing the objects' images to detect is located." << endl;
        cout << "VIDEO_PATH: path where the video is located." << endl;
        return 1;
    }
    String path = argv[1];
    VideoCapture cap(argv[2]);

    //load the images
    vector<String> filenames;
    glob(path, filenames);
    for (String fn : filenames)
    {
        objects.push_back(imread(fn));
    }
    for (Mat object : objects)
    {
        if (object.empty()) {
            cout << "File not found" << endl;
            return -1;
        }
    }

    // assign a random color to each object
    vector<Scalar> colors(4);
    for (int i = 0; i < 4; i++)
    {
        colors[i] = Scalar(rand() % 255, rand() % 255, rand() % 255);
    }

    //extract keypoints and descriptors
    vector<vector<KeyPoint>> objKeypoints;
    vector<Mat> objDescriptors;
    for (int i = 0; i < objects.size(); i++)
    {
        objKeypoints.push_back(extractKeypoints(objects[i]));
        objDescriptors.push_back(extractDescriptors(objects[i], objKeypoints[i]));
    }

    if (cap.isOpened())
    {
        //get the first frame and check if it is empty
        Mat firstFrame;
        cap >> firstFrame;
        if (firstFrame.empty())
        {
            cout << "Empty video!" << endl;
            return -1;;
        }
        resize(firstFrame, firstFrame, Size(firstFrame.cols / 2, firstFrame.rows / 2));

        //extract the keypoints and the descriptors from the first frame
        vector<KeyPoint> frameKeypoints;
        Mat frameDescriptors;
        frameKeypoints = extractKeypoints(firstFrame);
        frameDescriptors = extractDescriptors(firstFrame, frameKeypoints);

        //find matches in the first frame
        vector<vector<DMatch>> dmatches = findMatches(objDescriptors, frameDescriptors);

        //extract the homographies and the masks
        vector<vector<uint8_t>> masks;
        vector<Mat> homographies = findPointsHomographies(objKeypoints, frameKeypoints, dmatches, masks);

        //find the good matches using the extracted masks
        vector<vector<Point2f>> framePoints;
        vector<vector<DMatch>> goodMathces;
        for (int i = 0; i < objects.size(); ++i)
        {
            vector<DMatch> good_matches;
            vector<Point2f> frame_points;
            for (int j = 0; j < masks[i].size(); ++j)
            {
                if (masks[i][j])
                {
                    frame_points.push_back(frameKeypoints[dmatches[i][j].trainIdx].pt);
                    good_matches.push_back(dmatches[i][j]);
                }
            }
            framePoints.push_back(frame_points);
            goodMathces.push_back(good_matches);
        }

        //compute corners of the first frame
        vector<vector<Point2f>> objectCorners;
        for (Mat object : objects)
        {
            vector<Point2f> obj_corners = { Point2f(0, 0), Point2f(object.cols, 0), Point2f(object.cols, object.rows), Point2f(0, object.rows) };
            objectCorners.push_back(obj_corners);
        }
        vector<vector<Point2f>> scene_corners = computeRectCorners(objectCorners, homographies);

        //show the keypoints and the rectangular boundaries of the obejcts in the first frame
        Mat firsFrameDraw = firstFrame.clone();
        for (int i = 0; i < scene_corners.size(); ++i)
        {
            for (int j = 0; j < framePoints[i].size(); ++j)
            {
                circle(firsFrameDraw, framePoints[i][j], 3, colors[i], -1);
            }
            drawRect(firsFrameDraw, scene_corners[i], colors[i]);
        }
        imshow("Keypoints of first frame", firsFrameDraw);

        //show the matches with objects
        for (int i = 0; i < objects.size(); ++i)
        {
            Mat img_matches;
            drawMatches(objects[i], objKeypoints[i], firstFrame, frameKeypoints, goodMathces[i], img_matches);
            resize(img_matches, img_matches, Size(img_matches.cols / 2, img_matches.rows / 2));
            imshow("Match with object n. " + to_string(i + 1), img_matches);
        }

        //save the results to lead the next frame
        Mat frame_old = firstFrame.clone();
        vector<vector<Point2f>> oldRectPoints = scene_corners;

        while (true)
        {
            Mat frame;
            cap >> frame;
            //check if the video is finished
            if (frame.empty())
                break;
            resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
            Mat drawFrame = frame.clone();
            vector<vector<Point2f>> newPoints;
            vector<vector<Point2f>> newRectPoints;
            vector<vector<Point2f>> goodNewPoints;

            for (int i = 0; i < objects.size(); ++i)
            {
                //compute the optical flow
                vector<Point2f>  p1;
                vector<uchar> status;
                vector<float> err;
                TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
                calcOpticalFlowPyrLK(frame_old, frame, framePoints[i], p1, status, err, Size(7, 7), 3, criteria);
                newPoints.push_back(p1);

                //save the new points and draw them
                vector<Point2f> good_old;
                vector<Point2f> good_new;
                for (uint j = 0; j < framePoints[i].size(); j++)
                {
                    if (status[j] == 1)
                    {
                        good_old.push_back(framePoints[i][j]);
                        good_new.push_back(p1[j]);
                        circle(drawFrame, p1[j], 3, colors[i], -1);
                    }
                }
                goodNewPoints.push_back(good_new);

                //extract the homographies
                Mat H = findHomography(good_old, good_new);

                //compute the new rectangular boundaries
                vector<Point2f> new_rect_points;
                perspectiveTransform(oldRectPoints[i], new_rect_points, H);
                newRectPoints.push_back(new_rect_points);
                drawRect(drawFrame, new_rect_points, colors[i]);
            }

            //show the result
            imshow("Frame", drawFrame);
            int keyboard = waitKey(10);
            if (keyboard == 'q' || keyboard == 27)
                break;

            //save the result to lead the next frame
            frame_old = frame.clone();
            framePoints = goodNewPoints;
            oldRectPoints = newRectPoints;
        }
    }
    waitKey(0);
    return 0;
}