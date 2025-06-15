#include "main.hpp"

int findLabel(int *array, int n);
void sortArray(double arr[], int arr2[], int n);

template <typename T>
class List
{
public:
    virtual ~List() = default;
    virtual void push_back(T value) = 0;
    virtual void push_front(T value) = 0;
    virtual void insert(int index, T value) = 0;
    virtual void remove(int index) = 0;
    virtual T &get(int index) const = 0;
    virtual int length() const = 0;
    virtual void clear() = 0;
    virtual void print() const = 0;
    virtual void reverse() = 0;
    virtual int find(T value) = 0;
    virtual List<T> *subList(int start, int end) = 0;
    virtual void printStartToNullptr() = 0;
    virtual void printToEnd(int start, int end) const = 0;
};

template <typename T>
class LinkedList : public List<T>
{
private:
    class Node
    {
    public:
        T data;
        Node *next;

    public:
        Node(T data, Node *next = nullptr) : data(data), next(next) {}
    };
    Node *head;
    Node *tail;
    int size;
public:
    LinkedList(){
        size = 0;
        head = tail = nullptr;
    }
    ~LinkedList(){
        this->clear();
    }
    void push_back(T value){
        insert(size, value);
    }
    void push_front(T value){
        insert(0, value);
    }
    int find(T value){
        Node *current = head;
        int index = 0;
        while (current != nullptr){
            if (current->data == value){
                return index;
            }
            index += 1;
            current = current->next;
        }
        return -1;
    }
    void insert(int index, T value){
        if (index < 0 || index > size) {
            return;
        }
        if (index == 0){
            size += 1;
            Node *newNode = new Node(value, head);
            head = newNode;
            if (size == 1) {
                tail = newNode;
            }
            else if (size == 2){
                tail = newNode->next;
            }
        }
        else if (index == size){
            Node *newNode = new Node(value, nullptr);
            tail->next = newNode;
            tail = newNode;
            size += 1;
            if (size == 1) head = newNode;
        }
        else{
            Node *tmp = head;
            for (int i = 1; i < index; i++){
                tmp = tmp->next;
            }
            Node *newNode = new Node(value, tmp->next);
            tmp->next = newNode;
            size += 1;
        }
    }
    void remove(int index){
        if (index < 0 || index >= size)
            return;
        if (index == 0){
            Node *nodeDel = head;
            head = head->next;
            delete nodeDel;
            size -= 1;
            if (head == nullptr){
                tail = nullptr;
            }
        }
        else if (index == size - 1){
            Node *temp = head;
            while (temp->next != tail){
                temp = temp->next;
            }
            Node *nodeDel = temp->next;
            tail = temp;
            size -= 1;
            delete nodeDel;
            temp->next = nullptr;
        }
        else{
            Node *temp = head;
            for (int i = 0; i < index - 1; i++){
                temp = temp->next;
            }
            Node *nodeDel = temp->next;
            temp->next = nodeDel->next;
            delete nodeDel;
            size -= 1;
        }
    }

    T &get(int index) const{
        if (index >= this->size || index < 0)
            throw std::out_of_range("get(): Out of range");
        if (index == 0){
            if (head == nullptr){
                return tail->data;
            }
            return head->data;
        }
        else if (index == size - 1){
            return tail->data;
        }
        else{
            Node *temp = head->next;
            for (int i = 1; i < index; i++){
                temp = temp->next;
            }
            return temp->data;
        }
    }

    int length() const{
        return size;
    }

    void clear(){
        while (size > 0)
            remove(0);
        size = 0;
    }

    void print() const
    {
        if (size == 0)
            return;

        Node *temp = head;
        for (int i = 0; i < this->size; i++){
            if (i == this->size - 1)
                cout << temp->data;
            else
                cout << temp->data << " ";
            temp = temp->next;
        }
    }

    void printStartToNullptr(){
        Node *tmp = head;

        if (this->size == 1)
            return;
        while (tmp != nullptr){
            cout << tmp->data << ' ';
            tmp = tmp->next;
        }
        cout << '\n';
    }

    void reverse(){
        if (this->size == 1) return;
        Node *curr = head;
        tail = head;
        Node *prev = nullptr;
        Node *next = nullptr;

        while (curr != nullptr){
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        head = prev;
    }

    void printToEnd(int start, int end) const{
        Node *temp = head;
        for (int i = start; i < this->size && i < end; i++){
            if (i == this->size - 1 || i == end -1)
                cout << temp->data << endl;
            else
                cout << temp->data << " ";
        }
    }

    List<T> *subList(int start, int end){
        List<T> *result = new LinkedList<T>();

        if (start >= this->size)
            return result;

        if (this->size <= start)
            return result;

        if (end > size)
            end = size;

        Node *temp = head;
        for (int i = 0; i < start; i++)
            temp = temp->next;

        for (int i = start; i < end; i++)
        {
            result->push_back(temp->data);
            temp = temp->next;
        }

        return result;
    }
};

class Dataset
{
private:
    List<List<int> *> *data;
    List<string> *feature_col;
public:
    Dataset(){
        this->feature_col = new LinkedList<string>();
        this->data = new LinkedList<List<int> *>();
    }
    Dataset(List<List<int> *> *data, List<string> *feature_col){
        this->feature_col = feature_col;
        this->data = data;
    }
    ~Dataset(){
        for (int i = 0; i < data->length(); i++){
            data->get(i)->clear();
        }
        data->clear();
        delete data;
        delete feature_col;
    }
    Dataset(const Dataset &other){
        this->feature_col = new LinkedList<string>();
        this->data = new LinkedList<List<int> *>();

        for (int i = 0; i < other.feature_col->length(); i++){
            auto otherCol = other.feature_col->get(i);
            this->feature_col->push_back(otherCol);
        }

        int col = max(0, other.feature_col->length());
        for (int i = 0; i < other.data->length(); i++){
            auto otherData = other.data->get(i)->subList(0, col);
            this->data->push_back(otherData);
        }
    }

    Dataset &operator=(const Dataset &other){
        if (this == &other)
            return *this;

        this->~Dataset(); 

        if (other.data->length() == 0){
            data = new LinkedList<List<int> *>();
            feature_col = new LinkedList<string>();
            return *this;
        }

        int col = max(0, other.feature_col->length());
        this->data = new LinkedList<List<int> *>();
        int row = other.data->length();
        for (int i = 0; i < row; i++)
            data->push_back(other.data->get(i)->subList(0, col));

        this->feature_col = other.feature_col->subList(0, col);
        return *this;
    }
    List<List<int> *> *getData() const{
        return data;
    };
    bool loadFromCSV(const char *fileName){
        ifstream file(fileName);
        if (file.is_open()){
            string s;
            int number;
            file >> s;
            for (int j = 0; j < s.length(); j++){
                if (s[j] == ',')
                    s[j] = ' ';
            }

            stringstream ss(s);
            while (ss >> s)
                feature_col->push_back(s);
            while (file >> s){
                for (int j = 0; j < s.length(); j++){
                    if (s[j] == ',')
                        s[j] = ' ';
                }
                stringstream ss(s);
                List<int> *temp = new LinkedList<int>();
                while (ss >> number)
                    temp->push_back(number);
                data->push_back(temp);
            }
            return true;
        }
        return false;
    }
    void getShape(int &nRows, int &nCols) const{
        nRows = data->length();
        if (nRows < 1)
            nCols = 0;
        else
            nCols = feature_col->length();
    }

    void columns() const{
        unsigned int numCols = feature_col->length();
        for (int x = 0; x < numCols; x++){
            string label = feature_col->get(x);
            if (x < numCols - 1)
                cout << label << ' ';
            else
                cout << label;
        }
    }
    void printHead(int nRows = 5, int nCols = 5) const{
        if (nRows <= 0 || nCols <= 0)
            return;

        if (nRows > data->length())
            nRows = data->length();

        if (nCols > feature_col->length())
            nCols = feature_col->length();

        if (data->length() < 1){
            for (int i = 0; i < nCols; i++){
                string label = feature_col->get(i);
                if (i < nCols - 1)
                    cout << label << ' ';
                else
                    cout << label;
            }
            return;
        }
        for (int i = 0; i < nCols; i++){
            string label = feature_col->get(i);
            if (i < nCols - 1)
                cout << label << ' ';
            else
                cout << label << '\n';
        }
        for (int v = 0; v < nRows; v++){
            for (int w = 0; w < nCols; w++){
                int number = data->get(v)->get(w);

                if (w < nCols - 1)
                    cout << number << ' ';
                else
                    cout << number;
            }

            if (v < nRows - 1)
                cout << '\n';
        }
    }
    void printTail(int nRows = 5, int nCols = 5) const{
        if (nRows <= 0 || nCols <= 0)
            return;

        if (nRows > data->length())
            nRows = data->length();

        if (nCols > feature_col->length())
            nCols = feature_col->length();

        int colLength = feature_col->length();
        int rowLength = data->length();
        int startCol = colLength - nCols;
        int startRow = rowLength - nRows;

        if (data->length() < 1){
            for (int i = startCol; i < colLength; i++){
                string label = feature_col->get(i);
                if (i < colLength - 1)
                    cout << label << ' ';

                else
                    cout << label;
            }
            return;
        }

        for (int i = startCol; i < colLength; i++){
            string label = feature_col->get(i);
            if (i < colLength - 1)
                cout << label << ' ';

            else
                cout << label << '\n';
        }

        for (int i = startRow; i < rowLength; i++){
            for (int j = startCol; j < colLength; j++){
                int number = data->get(i)->get(j);

                if (j < colLength - 1)
                    cout << number << ' ';

                else
                    cout << number;
            }
            if (i < rowLength - 1) cout << '\n';
        }
    }
    bool drop(int axis = 0, int index = 0, std::string columns = ""){
        if (axis != 1 && axis != 0)
            return false;

        if (data->length() < 1){
            if (axis == 1){
                int found = feature_col->find(columns);
                if (found == -1) return false;
                else{
                    feature_col->remove(found);
                    return true;
                }
            }
            return false;
        }

        if (axis == 0){
            if (index >= data->length() || index < 0)
                return false;

            data->get(index)->clear();
            data->remove(index);
            return true;
        }

        if (axis == 1){
            int find = feature_col->find(columns);
            if (find == -1)
                return false;
            else{
                feature_col->remove(find);
                bool couldClear = true;
                for (int i = 0; i < data->length(); i++){
                    data->get(i)->remove(find);

                    if (data->get(i)->length() != 0)
                        couldClear = false;
                }
                if (couldClear)
                    data->clear();
                return true;
            }
        }
        return false;
    }
    Dataset extract(int startRow = 0, int endRow = -1, int startCol = 0, int endCol = -1) const {
        Dataset result;
        if (startRow >= data->length())
            return result;

        if (endRow == -1) {
            endRow = data->length() - 1;
        }
        else if (endRow >= data->length()){
            endRow = data->length() - 1;
        }

        if (endCol == -1){
            endCol = feature_col->length() - 1;
        }
        else if (endCol >= feature_col->length()){
            endCol = feature_col->length() - 1;
        }

        for (int i = startCol; i <= endCol; i++)
            result.feature_col->push_back(feature_col->get(i));

        if (startCol > endCol) return result;

        for (int i = startRow; i <= endRow; i++){
            List<int> *child = new LinkedList<int>();
            for (int j = startCol; j <= endCol; j++){
                int temp = this->data->get(i)->get(j);
                child->push_back(temp);
            }
            result.data->push_back(child);
        }

        return result;
    }

    double distanceEuclidean(const List<int> *x, const List<int> *y) const{
        double distance = 0.0;
        int size1 = x->length();
        int size2 = y->length();

        int index1 = 0;
        int index2 = 0;

        while (index2 < size2 || index1 < size1){
            int tem1 = 0;
            int tem2 = 0;
            if (index1 < size1) tem1 = x->get(index1);
            if (index2 < size2) tem2 = y->get(index2);

            distance += pow(abs(tem1 - tem2), 2);
            index1 += 1;
            index2 += 1;
        }
        return sqrt(distance);
    }

    Dataset predict(const Dataset &X_train, const Dataset &Y_train, const int k) const{
        Dataset result;
        if (k <= 0) return result;

        unsigned int xTrainSize = X_train.data->length();
        unsigned int yTrainSize = Y_train.data->length();
        unsigned int data_size = this->data->length();

        if (xTrainSize == 0 || yTrainSize == 0)return result;

        result.feature_col->push_back(Y_train.feature_col->get(0));
        int Length_train = min(xTrainSize, yTrainSize);
        int *label = new int[Length_train];
        double *distance = new double[Length_train];
        for (int i = 0; i < this->data->length(); i++){
            for (int j = 0; j < Length_train; j++){
                label[j] = Y_train.data->get(j)->get(0);
            }

            for (int j = 0; j < Length_train; j++){
                double dist = this->distanceEuclidean(data->get(i), X_train.data->get(j));
                distance[j] = dist;
            }

            sortArray(distance, label, Length_train);
            int predictRes = findLabel(label, min(Length_train, k));
            List<int> *temp1 = new LinkedList<int>();
            temp1->push_back(predictRes);
            result.data->push_back(temp1);
        }

        delete[] distance;
        delete[] label;
        return result;
    }
    double score(const Dataset &y_predict) const{
        if (y_predict.data->length() < 1)
            return -1;

        int data_size = this->data->length();
        if (data_size < 1)
            return -1;

        double matchCount = 0;
        int len = min(data_size, y_predict.data->length());
        for (int i = 0; i < len; i++){
            int temp1 = this->data->get(i)->get(0);
            int temp2 = y_predict.data->get(i)->get(0);
            if (temp1 == temp2)
                matchCount += 1;
        }
        return matchCount / data_size;
    }
};

class kNN{
private:
    int k;
    Dataset X_train;
    Dataset Y_train;
public:
    kNN(int k = 5) : k(k){};
    void fit(const Dataset &X_train, const Dataset &y_train){
        this->X_train = X_train;
        this->Y_train = y_train;
    }
    Dataset predict(const Dataset &X_test){
        return X_test.predict(this->X_train, this->Y_train, this->k);
    }
    double score(const Dataset &y_test, const Dataset &y_pred){
        return y_test.score(y_pred);
    }
};

void train_test_split(Dataset &X, Dataset &Y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &Y_train, Dataset &Y_test);
