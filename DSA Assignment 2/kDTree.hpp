#include "main.hpp"
#include "Dataset.hpp"

struct kDTreeNode {
    std::vector<int> data;
    kDTreeNode* left = nullptr;
    kDTreeNode* right = nullptr;

    explicit kDTreeNode(const std::vector<int>& data_) : data(data_) {}

    kDTreeNode(const kDTreeNode& other) : data(other.data) {
        if (other.left) {
            left = new kDTreeNode(*other.left);
        }
        if (other.right) {
            right = new kDTreeNode(*other.right);
        }
    }

    kDTreeNode& operator=(const kDTreeNode& other) {
        if (this != &other) {
            data = other.data;
            delete left;
            delete right;
            if (other.left) {
                left = new kDTreeNode(*other.left);
            } else {
                left = nullptr;
            }
            if (other.right) {
                right = new kDTreeNode(*other.right);
            } else {
                right = nullptr;
            }
        }
        return *this;
    }

    ~kDTreeNode() {
        delete left;
        delete right;
    }

    friend std::ostream& operator<<(std::ostream& os, const kDTreeNode& node) {
        os << "(";
        for (size_t i = 0; i < node.data.size(); ++i) {
            os << node.data[i];
            if (i != node.data.size() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};


// kDTree class
class kDTree
{
private:
    int k;            // Dimension of the kD-Tree
    kDTreeNode *root; // Root node of the kD-Tree

    // Private helper functions
    void inorderTraversal(kDTreeNode *root) const;                                                                         // Inorder traversal of the kD-Tree
    void preorderTraversal(kDTreeNode *root) const;                                                                        // Preorder traversal of the kD-Tree
    void postorderTraversal(kDTreeNode *root) const;                                                                       // Postorder traversal of the kD-Tree
    int height(kDTreeNode *root) const;                                                                                    // Calculate the height of the kD-Tree
    int nodeCount(kDTreeNode *root) const;                                                                                 // Count the number of nodes in the kD-Tree
    int leafCount(kDTreeNode *root) const;                                                                                 // Count the number of leaf nodes in the kD-Tree
    void insert(kDTreeNode *&root, const vector<int> &point, int depth);                                                   // Insert a new point into the kD-Tree
    kDTreeNode *findMin(kDTreeNode *root, int depth, int dim);                                                             // Find the node with the minimum value in a given dimension
    void remove(kDTreeNode *&root, const vector<int> &point, int depth);                                                   // Remove a point from the kD-Tree
    bool search(kDTreeNode *root, const vector<int> &point, int depth);                                                    // Search for a point in the kD-Tree
    int mergeSort(vector<vector<int>> &pointList, int left, int right, int depth);                                         // Merge sort helper function
    void buildTree(kDTreeNode *&root, vector<vector<int>> &pointList, int left, int right, int depth);                     // Build the kD-Tree from a list of points
    double distance_squared(const vector<int> &a, const vector<int> &b);                                                   // Calculate the squared Euclidean distance between two points
    void nearestNeighbour(kDTreeNode *root, const vector<int> &target, kDTreeNode *&best, int depth);                      // Find the nearest neighbor of a target point
    void kNearestNeighbour(kDTreeNode *root, const vector<int> &target, int k, vector<kDTreeNode *> &bestList, int depth); // Find the k nearest neighbors of a target point

public:
    kDTree(int k = 2); // Constructor with default dimension 2
    ~kDTree();         // Destructor

    const kDTree &operator=(const kDTree &other); // Assignment operator
    kDTree(const kDTree &other);                  // Copy constructor

    void inorderTraversal() const;   // Wrapper function for inorder traversal
    void preorderTraversal() const;  // Wrapper function for preorder traversal
    void postorderTraversal() const; // Wrapper function for postorder traversal
    int height() const;              // Wrapper function for height calculation
    int nodeCount() const;           // Wrapper function for node count
    int leafCount() const;           // Wrapper function for leaf count

    void insert(const vector<int> &point);                                                                     // Wrapper function for insertion
    void remove(const vector<int> &point);                                                                     // Wrapper function for removal
    bool search(const vector<int> &point);                                                                     // Wrapper function for search
    void buildTree(const vector<vector<int>> &pointList);                                                      // Wrapper function for building the kD-Tree
    void nearestNeighbour(const vector<int> &target, kDTreeNode *&best);                                       // Wrapper function for finding the nearest neighbor
    void addNearNeighbour(kDTreeNode *best, const vector<int> &target, vector<kDTreeNode *> &bestList, int k); // Helper function for adding nearest neighbors
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList);                  // Wrapper function for finding the k nearest neighbors
};

// kNN class
class kNN
{
private:
    int k;            // Number of nearest neighbors
    Dataset *X_train; // Training data
    Dataset *y_train; // Training labels
    kDTree *kdtree;   // kD-Tree for efficient nearest neighbor search
    int numClasses;   // Number of classes in the dataset

    int getNumClasses(const Dataset &y_train);              // Helper function to get the number of classes
    int getMajorityClass(const vector<int> &bestLabelList); // Helper function to get the majority class from a list of labels
    int getLabel(const vector<int> &point);                 // Helper function to get the label for a given point

public:
    kNN(int k = 5);                                             // Constructor with default k = 5
    void fit(const Dataset &X_train, const Dataset &y_train);   // Fit the kNN model to the training data
    Dataset predict(const Dataset &X_test);                     // Predict labels for the test data
    double score(const Dataset &y_test, const Dataset &y_pred); // Calculate the accuracy score
};