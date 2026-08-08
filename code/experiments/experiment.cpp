#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using namespace std::chrono_literals;
using json = nlohmann::json;

// ----------------------------
// CONFIG
// ----------------------------

static constexpr double TARGET_DISTANCE_M = 1.83;
static constexpr double FORWARD_SPEED = 0.25;
static constexpr double TURN_SPEED = 0.4;
static constexpr double STOP_DISTANCE_M = 1.2;
static constexpr double DEPTH_FOV_RATIO = 0.3;

static const std::string VLM_ENDPOINT = "http://localhost:5000/vlm";

// ----------------------------
// CURL HELPERS
// ----------------------------

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// ----------------------------
// NODE
// ----------------------------

class Go2Navigator : public rclcpp::Node
{
public:
    Go2Navigator() : Node("go2_yolo_vlm_nav")
    {
        rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10,
            std::bind(&Go2Navigator::rgbCallback, this, std::placeholders::_1));

        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10,
            std::bind(&Go2Navigator::depthCallback, this, std::placeholders::_1));

        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        start_time_ = now();

        RCLCPP_INFO(get_logger(), "Go2 C++ YOLO + VLM Navigator started");
    }

    void step()
    {
        if (rgb_frame_.empty() || depth_frame_.empty())
            return;

        double elapsed = (now() - start_time_).seconds();
        double distance_traveled = elapsed * FORWARD_SPEED;

        if (distance_traveled >= TARGET_DISTANCE_M)
        {
            stop();
            RCLCPP_INFO(get_logger(), "Reached target distance");
            rclcpp::shutdown();
            return;
        }

        double depth_ahead = getForwardDepth();

        std::string direction = "forward";

        if (depth_ahead < STOP_DISTANCE_M)
        {
            // --------------------------------
            // YOLO PLACEHOLDER (C++ inference)
            // --------------------------------
            bool yolo_has_box = false;
            double box_center_x = 0.0;

            // TODO: Replace with OpenCV DNN / TensorRT YOLO
            // If YOLO detects something, set yolo_has_box = true

            if (yolo_has_box)
            {
                direction = (box_center_x < rgb_frame_.cols / 2) ? "right" : "left";
            }
            else
            {
                direction = queryVLM(rgb_frame_);
            }
        }

        executeMotion(direction);
    }

private:
    // ----------------------------
    // ROS
    // ----------------------------

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

    rclcpp::Time start_time_;

    cv::Mat rgb_frame_;
    cv::Mat depth_frame_;

    // ----------------------------
    // CALLBACKS
    // ----------------------------

    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        rgb_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat depth_mm = cv_bridge::toCvCopy(msg)->image;
        depth_mm.convertTo(depth_frame_, CV_32F, 1.0 / 1000.0);  // mm → meters
    }

    // ----------------------------
    // DEPTH
    // ----------------------------

    double getForwardDepth()
    {
        int h = depth_frame_.rows;
        int w = depth_frame_.cols;

        int cx1 = static_cast<int>(w * (0.5 - DEPTH_FOV_RATIO / 2));
        int cx2 = static_cast<int>(w * (0.5 + DEPTH_FOV_RATIO / 2));
        int cy1 = static_cast<int>(h * 0.4);
        int cy2 = static_cast<int>(h * 0.8);

        cv::Mat roi = depth_frame_(cv::Range(cy1, cy2), cv::Range(cx1, cx2));
        cv::Mat mask = roi > 0;

        std::vector<float> values;
        roi.copyTo(values, mask);

        if (values.empty())
            return std::numeric_limits<double>::infinity();

        std::nth_element(values.begin(),
                         values.begin() + values.size() / 10,
                         values.end());

        return values[values.size() / 10];
    }

    // ----------------------------
    // VLM QUERY
    // ----------------------------

    std::string queryVLM(const cv::Mat& image)
    {
        std::vector<uchar> buffer;
        cv::imencode(".jpg", image, buffer);

        std::string img_b64 = base64Encode(buffer);

        json payload = {
            {"image", img_b64},
            {"prompt",
             "You are assisting a blind user using a robot dog. "
             "Respond with one word only: left, right, forward, or stop."}
        };

        CURL* curl = curl_easy_init();
        if (!curl)
            return "stop";

        std::string response;
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, VLM_ENDPOINT.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.dump().c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5);

        curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        auto j = json::parse(response, nullptr, false);
        std::string text = j.value("response", "");

        if (text.find("left") != std::string::npos) return "left";
        if (text.find("right") != std::string::npos) return "right";
        if (text.find("forward") != std::string::npos) return "forward";

        return "stop";
    }

    // ----------------------------
    // MOTION
    // ----------------------------

    void executeMotion(const std::string& direction)
    {
        geometry_msgs::msg::Twist cmd;

        if (direction == "forward")
            cmd.linear.x = FORWARD_SPEED;
        else if (direction == "left")
            cmd.angular.z = TURN_SPEED;
        else if (direction == "right")
            cmd.angular.z = -TURN_SPEED;

        cmd_pub_->publish(cmd);
    }

    void stop()
    {
        cmd_pub_->publish(geometry_msgs::msg::Twist());
    }

    // ----------------------------
    // BASE64
    // ----------------------------

    static std::string base64Encode(const std::vector<uchar>& data);
};
