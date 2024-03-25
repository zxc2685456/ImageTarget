#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui_c.h>
#include<iostream>
#include<opencv2/core/opengl.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include "BEBLID.h"


void ORB(cv::String imgPath);
cv::Mat createMask(cv::Size img_size, std::vector<cv::Point2f>& pts);
bool tracking = false;
bool good_detection = false;
int main()
{
    ORB("laker.png");
}

void ORB(cv::String imgPath)
{
    cv::VideoCapture capture(1);

    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 50);

    std::vector<cv::Point2f>prev_keypoints, curr_keyPoints;
    std::vector<cv::Point2f>prev_corners, curr_corners;
    cv::Mat prev_gray, curr_gray;
    std::vector<cv::Mat>prevPyr, nextPyr;
    std::vector<unsigned char> track_status;
    cv::Scalar green = cv::Scalar(0, 255, 0);
    cv::Scalar red = cv::Scalar(0, 0, 255);
    cv::Scalar blue = cv::Scalar(255, 0, 0);

    /* Keypoints & descriptor detection. */
    cv::Mat imgQ = cv::imread(imgPath), imgT, imgM, imgvedio;
    capture >> imgT;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(500);
    std::vector<cv::KeyPoint> keypointsQ, keypointsT;
    cv::Mat descriptorQ, descriptorT;
    cv::Mat scene_mask;
    detector->detectAndCompute(imgQ, cv::Mat(), keypointsQ, descriptorQ);
    cv::Mat H_latest = cv::Mat::eye(3, 3, CV_32F);


    while (capture.read(imgT))
    {
        scene_mask = cv::Mat::zeros(imgT.rows, imgT.cols, CV_8UC1);
        imgT.copyTo(imgvedio);
        if (tracking == false)
        {
            bool good_detection = false;
            //capture >> imgT;
            //imgT.copyTo(imgvedio);
            detector->detectAndCompute(imgT, cv::Mat(), keypointsT, descriptorT);
            
            //if (descriptorQ.empty() || descriptorT.empty())
            //    continue;

            //auto detector = BEBLID::create(256, 0.75);
            //auto detector = cv::xfeatures2d::BEBLID::create(256, 0.75);
            cv::Mat descriptor1, descriptor2;
            detector->compute(imgQ, keypointsQ, descriptor1);
            detector->compute(imgT, keypointsT, descriptor2);

            /* Matching & filtering. */
            cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(6, 12, 1));
            //cv::BFMatcher matcher(cv::NORM_HAMMING2, true);

            std::vector<cv::DMatch> matches;

            matcher.match(descriptorQ, descriptorT, matches);

            //std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) {return a.distance < b.distance; });
            //std::vector<cv::DMatch> goodMatches = std::vector<cv::DMatch>(matches.begin() + 15, matches.begin() + 40);


            double max_dist = 0; double min_dist = 100;
            std::vector<cv::DMatch> goodMatches;
            for (int i = 0; i < matches.size(); i++)
            {
                double dist = matches[i].distance;
                if (dist < min_dist)
                    min_dist = dist;
                if (dist > max_dist)
                    max_dist = dist;
                if (matches[i].distance < 0.4 * max_dist)
                    goodMatches.push_back(matches[i]);
            } 

            /* Matches drawing for demo purpose. */

            cv::drawMatches(imgQ, keypointsQ, imgT, keypointsT, goodMatches, imgM);
            if (goodMatches.size() >= 10)
            {
                std::vector<cv::Point2f> pointsQ, pointsT;

                for (const auto& match : goodMatches)
                {
                    pointsQ.push_back(keypointsQ[match.queryIdx].pt);                                           
                    pointsT.push_back(keypointsT[match.trainIdx].pt);
                }



                if (pointsT.size() > 20)
                {
                    cv::Mat H = findHomography(pointsQ, pointsT, cv::RANSAC, 5);

                    std::array<cv::Point2f, 4> cornersQ = { cv::Point(0, 0), cv::Point(imgQ.cols, 0), cv::Point(imgQ.cols, imgQ.rows), cv::Point(0, imgQ.rows) };
                    std::array<cv::Point2f, 4> cornersT;

                    cv::perspectiveTransform(cornersQ, cornersT, H);
                    std::cout << "ORB H " << H << std::endl;
                    std::vector<cv::Point2f> square;
                    square.clear();
                    for (int i = 0; i < 4; i++)
                    {
                        cv::Point2f pt1 = cv::Point2f(imgQ.cols, 0) + cornersT[i];
                        cv::Point2f pt2 = cv::Point2f(imgQ.cols, 0) + cornersT[(i + 1) % 4];
                        cv::line(imgM, pt1, pt2, cv::Scalar(0, 0, 255), 2);
                        square.push_back(pt1);
                    }

                    prev_keypoints.clear();
                    prev_corners.clear();
                    cv::cvtColor(imgT, prev_gray, CV_BGR2GRAY);
                    for (int j = 0; j < 4; j++)
                    {
                        prev_corners.push_back(cornersT[j]);
                    }
                    //if (square[1].x - square[0].x > 200 && square[3].y - square[0].y >150 && square[3].x - square[0].x <50 && (square[3].y - square[0].y) - (square[2].y - square[1].y) < 30)
                    //    tracking = true;
                    cv::Mat mask = createMask(imgT.size(), prev_corners);
                    cv::goodFeaturesToTrack(prev_gray, prev_keypoints, 80, 0.15, 5, mask);
                    prevPyr.clear();
                    track_status.clear();
                    tracking = true;

                }
            }

        }
        else
        {
            int num = 0;
            for (int i = 0; i < 4; i++)
            {
                if (prev_corners[i].x <= 0 || prev_corners[i].x >= imgT.cols || prev_corners[i].y <= 0 || prev_corners[i].y >= imgT.rows)
                {
                    num++;
                }
            }
            if (num != 0)
            {
                cv::Mat mask = createMask(imgT.size(), prev_corners);
                cv::goodFeaturesToTrack(prev_gray, prev_keypoints, 80, 0.15, 5, mask);
            }

            if (prev_keypoints.size() >= 20)
            {
                cv::cvtColor(imgT, curr_gray, CV_BGR2GRAY);

                std::vector<float> err;
                if (prevPyr.empty())
                {
                    cv::buildOpticalFlowPyramid(prev_gray, prevPyr, cv::Size(21, 21), 3, true);
                }
                cv::buildOpticalFlowPyramid(curr_gray, nextPyr, cv::Size(21, 21), 3, true);
                cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, prev_keypoints, curr_keyPoints, track_status, err, cv::Size(21, 21), 3);

                int tr_num = 0;
                std::vector<cv::Point2f> trackedPrePts;
                std::vector<cv::Point2f> trackedPts;
                for (size_t i = 0; i < track_status.size(); i++)
                {
                    if (track_status[i] && prev_keypoints.size() > i && cv::norm(curr_keyPoints[i] - prev_keypoints[i]) <= 15)
                    {
                        tr_num++;
                        trackedPrePts.push_back(prev_keypoints[i]);
                        trackedPts.push_back(curr_keyPoints[i]);
                    }
                }
                if (tr_num >= 15)
                {
                    cv::Mat homograyphyMat = cv::findHomography(trackedPrePts, trackedPts, cv::RANSAC, 5);

                    if (cv::countNonZero(homograyphyMat) != 0)
                    {
                        curr_corners.clear();
                        cv::perspectiveTransform(prev_corners, curr_corners, homograyphyMat);
                        std::cout << "Optical Flow H " << homograyphyMat << std::endl;
                        float hDet = abs(cv::determinant(homograyphyMat));
                        if (hDet < 100 && hDet > 0.05)
                        {
                            H_latest = homograyphyMat;
                            good_detection = true;
                        }

                        cv::swap(prev_gray, curr_gray);
                        prevPyr.swap(nextPyr);
                        prev_keypoints = trackedPts;
                        prev_corners = curr_corners;

                        if (prev_corners.size() >= 4)
                        {
                            cv::line(imgT, prev_corners[0], prev_corners[1], cv::Scalar(255, 0, 0, 255), 2);
                            cv::line(imgT, prev_corners[1], prev_corners[2], cv::Scalar(0, 255, 0, 255), 2);
                            cv::line(imgT, prev_corners[2], prev_corners[3], cv::Scalar(0, 0, 255, 255), 2);
                            cv::line(imgT, prev_corners[3], prev_corners[0], cv::Scalar(0, 0, 0, 255), 2);

                            cv::FileStorage fs("out_camera_data.xml", cv::FileStorage::READ);
                            cv::Mat intrinsics, distortion;
                            fs["Camera_Matrix"] >> intrinsics;
                            fs["Distortion_Coefficients"] >> distortion;

                            std::vector<cv::Point3f> objectPoints;

                            objectPoints.push_back(cv::Point3f(-1, 1, 0));
                            objectPoints.push_back(cv::Point3f(1, 1, 0));
                            objectPoints.push_back(cv::Point3f(1, -1, 0));
                            objectPoints.push_back(cv::Point3f(-1, -1, 0));
                            cv::Mat objectPointsMat(objectPoints);

                            cv::Mat rvec;
                            cv::Mat tvec;
                            cv::solvePnP(objectPointsMat, curr_corners, intrinsics, distortion, rvec, tvec);

                            std::vector<cv::Point3f> line3dx = { {0, 0, 0}, {2, 0, 0} };      
                            std::vector<cv::Point3f> line3dy = { {0, 0, 0}, {0, 2, 0} };
                            std::vector<cv::Point3f> line3dz = { {0, 0, 0}, {0, 0, 2} };

                            std::vector<cv::Point2f> line2dx;
                            std::vector<cv::Point2f> line2dy;
                            std::vector<cv::Point2f> line2dz;

                            projectPoints(line3dx, rvec, tvec, intrinsics, distortion, line2dx);
                            projectPoints(line3dy, rvec, tvec, intrinsics, distortion, line2dy);
                            projectPoints(line3dz, rvec, tvec, intrinsics, distortion, line2dz);

                            line(imgT, line2dx[0], line2dx[1], red, 3);
                            line(imgT, line2dy[0], line2dy[1], blue, 3);
                            line(imgT, line2dz[0], line2dz[1], green, 3);

                            std::cout << "rvec" << rvec << std::endl;
                            std::cout << "tvec" << tvec << std::endl;
                            //std::cout << "H" << homograyphyMat << std::endl;
                            
                            cv::imshow("iomg2", imgT);
                        }
                    }
                }
                if (tr_num < 15)
                {
                    tracking = false;
                }

            }

            else
                tracking = false;

        }
        cv::imshow("result image", imgM);
        cv::waitKey(5);
    }

}

cv::Mat createMask(cv::Size img_size, std::vector<cv::Point2f>& pts)
{
    cv::Mat mask(img_size, CV_8UC1);
    float zero = 0;
    mask = zero;
    // ax+by+c=0
    float a[4];
    float b[4];
    float c[4];

    a[0] = pts[3].y - pts[0].y;
    a[1] = pts[2].y - pts[1].y;
    a[2] = pts[1].y - pts[0].y;
    a[3] = pts[2].y - pts[3].y;

    b[0] = pts[0].x - pts[3].x;
    b[1] = pts[1].x - pts[2].x;
    b[2] = pts[0].x - pts[1].x;
    b[3] = pts[3].x - pts[2].x;

    c[0] = pts[0].y * pts[3].x - pts[3].y * pts[0].x;
    c[1] = pts[1].y * pts[2].x - pts[2].y * pts[1].x;
    c[2] = pts[0].y * pts[1].x - pts[1].y * pts[0].x;
    c[3] = pts[3].y * pts[2].x - pts[2].y * pts[3].x;

    float max_x, min_x, max_y, min_y;
    max_x = 0;
    min_x = img_size.width;
    max_y = 0;
    min_y = img_size.height;

    int i;
    for (i = 0; i < 4; i++) {
        if (pts[i].x > max_x)
            max_x = pts[i].x;
        if (pts[i].x < min_x)
            min_x = pts[i].x;
        if (pts[i].y > max_y)
            max_y = pts[i].y;
        if (pts[i].y < min_y)
            min_y = pts[i].y;
    }
    if (max_x >= img_size.width)
        max_x = img_size.width - 1;
    if (max_y >= img_size.height)
        max_y = img_size.height - 1;
    if (min_x < 0)
        min_x = 0;
    if (min_y < 0)
        min_y = 0;

    unsigned char* ptr = mask.data;
    int x, y;
    int offset;
    float val[4];
    for (y = min_y; y <= max_y; y++) {
        offset = y * img_size.width;
        for (x = min_x; x <= max_x; x++) {
            for (i = 0; i < 4; i++) {
                val[i] = a[i] * x + b[i] * y + c[i];
            }
            if (val[0] * val[1] <= 0 && val[2] * val[3] <= 0)
                *(ptr + offset + x) = 255;
        }
    }
    return mask;
}