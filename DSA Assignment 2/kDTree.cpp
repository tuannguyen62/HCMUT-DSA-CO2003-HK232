#include "kDTree.hpp"

// Constructor that initializes the tree dimension and sets the root to nullptr using an initializer list
kDTree::kDTree(int k) : k(k), root(nullptr) {}

// Destructor that handles the deletion of the root node if it exists
kDTree::~kDTree() {
    delete root;
    root = nullptr; // Ensure the root pointer is set to nullptr after deletion
}

// Copy assignment operator that performs a deep copy to prevent issues with shared memory
const kDTree &kDTree::operator=(const kDTree &other) {
    if (this != &other) { // Protect against self-assignment
        k = other.k; // Copy the dimension
        delete root; // Delete the current tree to prevent memory leaks
        root = other.root ? new kDTreeNode(*other.root) : nullptr; // Deep copy of the tree if not null
    }
    return *this; // Return a reference to the current instance
}

// Copy constructor that initializes the dimension and root, and performs a deep copy if necessary
kDTree::kDTree(const kDTree &other) : k(other.k), root(nullptr) {
    if (other.root) {
        root = new kDTreeNode(*other.root); // Perform a deep copy of the root
    }
}



// Perform an inorder traversal on the subtree rooted at the given node
void kDTree::inorderTraversal(kDTreeNode *node) const {
    if (!node) return;  // Base case: stop if the node is nullptr
    inorderTraversal(node->left);  // Recursively traverse the left subtree
    cout << *node;  // Visit the current node
    if (node->right) cout << " ";  // Print a space before going to the right subtree if it exists
    inorderTraversal(node->right);  // Recursively traverse the right subtree
}

// Public interface for performing an inorder traversal starting from the root
void kDTree::inorderTraversal() const {
    inorderTraversal(root);  // Start traversal from the root
}

// Perform a preorder traversal on the subtree rooted at the given node
void kDTree::preorderTraversal(kDTreeNode *node) const {
    if (!node) return;  // Base case: stop if the node is nullptr
    cout << " " << *node;  // Visit the current node and print it with a leading space
    preorderTraversal(node->left);  // Recursively traverse the left subtree
    preorderTraversal(node->right);  // Recursively traverse the right subtree
}

// Public interface for performing a preorder traversal starting from the root
void kDTree::preorderTraversal() const {
    if (!root) return;  // Check if the root is nullptr before starting
    cout << *root;  // Print the root node first
    preorderTraversal(root->left);  // Then traverse the left subtree
    preorderTraversal(root->right);  // Then traverse the right subtree
}

// Perform a postorder traversal on the subtree rooted at the given node
void kDTree::postorderTraversal(kDTreeNode *node) const {
    if (!node) return;  // Base case: stop if the node is nullptr
    postorderTraversal(node->left);  // Recursively traverse the left subtree
    postorderTraversal(node->right);  // Recursively traverse the right subtree
    cout << *node << " ";  // Visit the current node and print it with a trailing space
}

// Perform a postorder traversal on the entire tree starting from the root
void kDTree::postorderTraversal() const {
    if (!root) return;  // Base case: if root is nullptr, terminate the traversal
    postorderTraversal(root->left);  // Recursively traverse the left subtree
    postorderTraversal(root->right);  // Recursively traverse the right subtree
    cout << *root;  // Process the root node last
}

// Calculate the height of the tree starting from the given node
int kDTree::height(kDTreeNode *node) const {
    if (!node) return 0;  // If node is nullptr, the height is 0
    // Recursively find the maximum height between left and right subtrees and add 1 for the current node
    return 1 + max(height(node->left), height(node->right));
}

// Public interface to get the height of the entire tree
int kDTree::height() const {
    return height(root);  // Calculate height starting from the root
}

// Calculate the number of nodes in the tree starting from the given node
int kDTree::nodeCount(kDTreeNode *node) const {
    if (!node) return 0;  // If node is nullptr, there are no nodes
    // Sum the count of nodes in left and right subtrees, adding 1 for the current node
    return 1 + nodeCount(node->left) + nodeCount(node->right);
}

// Public interface to count the total number of nodes in the tree
int kDTree::nodeCount() const {
    return nodeCount(root);  // Start node count from the root
}

// Calculate the number of leaf nodes starting from the given node
int kDTree::leafCount(kDTreeNode *node) const {
    if (!node) return 0;  // If node is nullptr, there are no leaves
    if (!node->left && !node->right) return 1;  // If no children, it's a leaf node
    // Recursively count leaf nodes in both left and right subtrees
    return leafCount(node->left) + leafCount(node->right);
}


// Returns the count of leaf nodes in the tree
int kDTree::leafCount() const {
    return leafCount(root);  // Use the recursive helper to count leaves starting from the root
}

// Inserts a new point into the kDTree, respecting the tree's dimensional rules
void kDTree::insert(kDTreeNode *&node, const vector<int> &point, int depth) {
    if (point.size() != k) return;  // Ensure the point has the correct dimensionality
    if (!node) {  // If the node is nullptr, insert the new point here
        node = new kDTreeNode(point);
        return;
    }
    // Determine the current dimension to compare based on the depth
    int currentDim = depth % k;
    if (point[currentDim] < node->data[currentDim]) {
        insert(node->left, point, depth + 1);  // Go left if the point is less in the current dimension
    } else {
        insert(node->right, point, depth + 1);  // Otherwise, go right
    }
}

// Public interface to insert a point into the tree
void kDTree::insert(const vector<int> &point) {
    insert(root, point, 0);  // Start insertion from the root
}

// Helper function to find the minimum node in the specified dimension
kDTreeNode* kDTree::findMin(kDTreeNode *node, int depth, int dim) {
    if (!node) return nullptr;  // Return nullptr if the node is null
    kDTreeNode *min = node;  // Start with the current node as the minimum
    if (depth % k == dim) {  // If we're comparing the correct dimension
        if (node->left) {  // Only go left if there's a left child
            kDTreeNode *leftMin = findMin(node->left, depth + 1, dim);
            if (leftMin && leftMin->data[dim] < min->data[dim]) min = leftMin;
        }
    } else {  // Otherwise, check both sides
        kDTreeNode *leftMin = findMin(node->left, depth + 1, dim);
        kDTreeNode *rightMin = findMin(node->right, depth + 1, dim);
        if (leftMin && leftMin->data[dim] < min->data[dim]) min = leftMin;
        if (rightMin && rightMin->data[dim] < min->data[dim]) min = rightMin;
    }
    return min;  // Return the node with the minimum value in the specified dimension
}

// Removes a point from the kDTree
void kDTree::remove(kDTreeNode *&node, const vector<int> &point, int depth) {
    if (!node) return;  // If the node is nullptr, just return
    if (node->data == point) {  // If the node matches the point to remove
        if (!node->left && !node->right) {  // Node is a leaf
            delete node;  // Delete the node
            node = nullptr;  // Set the pointer to nullptr
        } else if (node->right) {  // If there is a right child, find the minimum node in the right subtree
            kDTreeNode *target = findMin(node->right, depth + 1, depth % k);
            node->data = target->data;  // Replace the node's data with the minimum
            remove(node->right, target->data, depth + 1);  // Recursively remove the target node
        } else {  // No right child, but there is a left child
            kDTreeNode *target = findMin(node->left, depth + 1, depth % k);
            node->data = target->data;
            remove(node->left, target->data, depth + 1);
        }
    } else {  // If the node does not match
        int currentDim = depth % k;
        if (point[currentDim] < node->data[currentDim]) {  // Determine which side to recurse on
            remove(node->left, point, depth + 1);
        } else {
            remove(node->right, point, depth + 1);
        }
    }
}

// Public interface to remove a point from the kDTree
void kDTree::remove(const vector<int> &point) {
    remove(root, point, 0);  // Start removal from the root node
}

// Searches for a point in the kDTree using a recursive approach
bool kDTree::search(kDTreeNode *node, const vector<int> &point, int depth) {
    if (!node) return false;  // Return false if node is nullptr (base case)
    if (node->data == point) return true;  // Return true if the point is found
    int currentDim = depth % k;  // Calculate current dimension based on depth
    if (point[currentDim] < node->data[currentDim]) {
        return search(node->left, point, depth + 1);  // Search left subtree
    } else {
        return search(node->right, point, depth + 1);  // Search right subtree
    }
}

// Public interface to search for a point in the tree
bool kDTree::search(const vector<int> &point) {
    return search(root, point, 0);  // Start search from the root
}

// Sorts a list of points using the merge sort algorithm with dimension-specific comparisons
int kDTree::mergeSort(vector<vector<int>> &pointList, int left, int right, int depth) {
    if (left >= right) return left;  // Base case: if there is only one element, return
    depth %= k;  // Ensure depth is within the correct range of dimensions
    int mid = (left + right) / 2;  // Calculate the middle index for splitting
    mergeSort(pointList, left, mid, depth);  // Recursively sort the left half
    mergeSort(pointList, mid + 1, right, depth);  // Recursively sort the right half

    // Temporary array for merging sorted halves
    vector<vector<int>> temp(right - left + 1);
    int i = left, j = mid + 1, idx = 0;

    // Merge the sorted halves into a single sorted sequence
    while (i <= mid && j <= right) {
        if (pointList[i][depth] < pointList[j][depth]) {
            temp[idx++] = pointList[i++];
        } else {
            temp[idx++] = pointList[j++];
        }
    }
    while (i <= mid) {
        temp[idx++] = pointList[i++];
    }
    while (j <= right) {
        temp[idx++] = pointList[j++];
    }

    // Copy the sorted sequence back to the original list
    for (int i = left; i <= right; i++) {
        pointList[i] = temp[i - left];
    }
    return mid;  // Return the index of the middle element
}

// Builds the kDTree from a list of points by sorting them and recursively constructing nodes
void kDTree::buildTree(kDTreeNode *&node, vector<vector<int>> &pointList, int left, int right, int depth) {
    if (left > right) return;  // Base case: no points to build from
    if (left == right) {
        node = new kDTreeNode(pointList[left]);  // Create a node from a single point
        return;
    }
    int mid = mergeSort(pointList, left, right, depth);  // Sort points and find the middle index
    node = new kDTreeNode(pointList[mid]);  // Create a node from the midpoint
    buildTree(node->left, pointList, left, mid - 1, depth + 1);  // Recursively build the left subtree
    buildTree(node->right, pointList, mid + 1, right, depth + 1);  // Recursively build the right subtree
}

// Public interface to build the kDTree from an external list of points
void kDTree::buildTree(const vector<vector<int>> &pointList) {
    vector<vector<int>> tempList = pointList;  // Copy the list to modify it during sorting
    buildTree(root, tempList, 0, tempList.size() - 1, 0);  // Start building the tree from the root
}

// Calculates the squared distance between two points in k-dimensional space
double kDTree::distance_squared(const vector<int> &a, const vector<int> &b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += pow(a[i] - b[i], 2);  // Sum the squared differences of coordinates
    }
    return sum;  // Return the sum of squared differences
}

// Recursively finds the nearest neighbour of a target point in the kDTree
void kDTree::nearestNeighbour(kDTreeNode *node, const vector<int> &target, kDTreeNode *&best, int depth) {
    if (!node) return;  // Base case: node is nullptr
    if (node->data == target) {
        best = node;  // Target is exactly at the current node
        return;
    }
    int currentDim = depth % k;  // Determine the current dimension based on depth
    // Recurse into the tree based on the current dimension comparison
    kDTreeNode *next = target[currentDim] < node->data[currentDim] ? node->left : node->right;
    nearestNeighbour(next, target, best, depth + 1);

    // Check if the current node is closer to the target than the current best
    if (!best || distance_squared(node->data, target) < distance_squared(best->data, target)) {
        best = node;
    }
    // Check the opposite branch if it might contain a closer point
    double distToPlane = pow(target[currentDim] - node->data[currentDim], 2);
    if (distance_squared(best->data, target) > distToPlane) {
        nearestNeighbour(target[currentDim] < node->data[currentDim] ? node->right : node->left, target, best, depth + 1);
    }
}

// Public interface for finding the nearest neighbour
void kDTree::nearestNeighbour(const vector<int> &target, kDTreeNode *&best) {
    nearestNeighbour(root, target, best, 0);  // Start the search from the root
}

// Adds the nearest neighbour to the list, maintaining the list's order and size
void kDTree::addNearNeighbour(kDTreeNode *candidate, const vector<int> &target, vector<kDTreeNode *> &bestList, int k) {
    double candidateDist = distance_squared(candidate->data, target);
    bool inserted = false;

    for (auto it = bestList.begin(); it != bestList.end(); ++it) {
        if (candidateDist < distance_squared((*it)->data, target)) {
            bestList.insert(it, candidate); // Insert before the first element greater than the candidate
            inserted = true;
            break;
        }
    }

    if (!inserted && bestList.size() < k) {
        bestList.push_back(candidate); // Add to the end if not inserted and there is space
    }

    if (bestList.size() > k) {
        bestList.pop_back(); // Maintain the size limit
    }
}

// Searches for k nearest neighbours starting from a given root
void kDTree::kNearestNeighbour(kDTreeNode *node, const vector<int> &target, int k, vector<kDTreeNode *> &bestList, int depth) {
    if (!node) return;  // Base case: no node to process

    // Recurse depending on comparison in the current dimension
    int currentDim = depth % this->k;
    kDTreeNode *first = target[currentDim] < node->data[currentDim] ? node->left : node->right;
    kDTreeNode *second = target[currentDim] < node->data[currentDim] ? node->right : node->left;

    kNearestNeighbour(first, target, k, bestList, depth + 1);
    addNearNeighbour(node, target, bestList, k); // Check and add the current node to the list

    // Check the opposite side if needed
    double distToPlane = pow(target[currentDim] - node->data[currentDim], 2);
    if (bestList.size() < k || distance_squared(bestList.back()->data, target) > distToPlane) {
        kNearestNeighbour(second, target, k, bestList, depth + 1);
    }
}

// Public interface for k-nearest neighbour search
void kDTree::kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList) {
    kNearestNeighbour(root, target, k, bestList, 0);  // Begin search from the root
}

// Implementation of k-Nearest Neighbors machine learning algorithm
kNN::kNN(int k) : k(k) {}  // Constructor initializes k

// Trains the model on the training dataset
void kNN::fit(const Dataset &X_train, const Dataset &y_train) {
    int nRows, nCols;
    this->X_train = new Dataset(X_train);
    this->X_train->getShape(nRows, nCols);  // Retrieve dimensions of the training data
    this->y_train = new Dataset(y_train);
    this->numClasses = getNumClasses(y_train);  // Determine the number of classes in the target
    this->kdtree = new kDTree(nCols);  // Create a new kDTree with the number of features as dimensions

    // Prepare data points for the kDTree
    vector<vector<int>> pointList;
    for (auto &row : this->X_train->data) {
        vector<int> point(row.begin(), row.end());  // Convert rows to points
        pointList.push_back(point);
    }
    this->kdtree->buildTree(pointList);  // Build the kDTree with the list of points
}

// Predicts labels for a test dataset
Dataset kNN::predict(const Dataset &X_test) {
    Dataset y_pred;
    y_pred.columnName = this->y_train->columnName;  // Set the predicted dataset's column name
    int nRows, nCols;
    X_test.getShape(nRows, nCols);  // Retrieve dimensions of the test data
    for (auto &row : X_test.data) {
        vector<int> point(row.begin(), row.end());  // Convert test data rows to points
        vector<kDTreeNode *> bestList;
        this->kdtree->kNearestNeighbour(point, this->k, bestList);  // Find k-nearest neighbours

        // Collect the labels of the k-nearest neighbours
        vector<int> bestLabelList;
        for (auto *best : bestList) {
            bestLabelList.push_back(getLabel(best->data));
        }
        int bestClass = getMajorityClass(bestLabelList);  // Determine the majority class among neighbours
        y_pred.data.push_back({bestClass});  // Store the predicted class
    }
    return y_pred;  // Return the dataset of predicted labels
}

// Determines the maximum class label in the training dataset, which indicates the number of classes
int kNN::getNumClasses(const Dataset &y_train) {
    int nRows, nCols;
    y_train.getShape(nRows, nCols);  // Get the shape of the dataset
    int maxLabel = 0;  // Initialize the maximum label found in the dataset
    for (const auto& row : y_train.data) {  // Iterate over each row in the dataset
        for (const auto& ele : row) {  // Iterate over each element in the row
            if (ele > maxLabel) {  // Update maxLabel if a larger element is found
                maxLabel = ele;
            }
        }
    }
    return maxLabel + 1;  // Return the count of classes, assuming labels start from 0
}

// Returns the class with the majority vote from nearest neighbors
int kNN::getMajorityClass(const vector<int> &bestLabelList) {
    vector<int> labelCount(this->numClasses, 0);  // Vector to count occurrences of each label
    for (int label : bestLabelList) {  // Count each label in the list
        labelCount[label]++;
    }
    int maxCount = 0, maxClass = 0;
    for (int i = 0; i < labelCount.size(); i++) {  // Iterate through counts to find the majority
        if (labelCount[i] > maxCount) {  // Update maxCount and maxClass if a higher count is found
            maxCount = labelCount[i];
            maxClass = i;
        }
    }
    return maxClass;  // Return the class with the most occurrences
}

int kNN::getLabel(const vector<int> &point) {
    auto i = this->X_train->data.begin();
    auto j = this->y_train->data.begin();
    while (i != this->X_train->data.end() && j != this->y_train->data.end()) {
        // Convert list to vector for comparison
        vector<int> temp(i->begin(), i->end());
        if (temp == point) {  // Now we are comparing two vectors
            return j->front();
        }
        ++i;
        ++j;
    }
    return -1;  // Return -1 if no label is found
}
double kNN::score(const Dataset &y_test, const Dataset &y_pred) {
    int nRows, nCols;
    y_test.getShape(nRows, nCols);  // Properly call getShape with two arguments
    int correct = 0;
    auto i = y_test.data.begin();
    auto j = y_pred.data.begin();
    while (i != y_test.data.end() && j != y_pred.data.end()) {
        if ((*i).front() == (*j).front()) {
            correct++;
        }
        ++i;
        ++j;
    }
    return static_cast<double>(correct) / nRows;  // Calculate accuracy as a percentage
}

