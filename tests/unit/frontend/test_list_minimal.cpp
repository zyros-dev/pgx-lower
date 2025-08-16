#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"
#include <iostream>

TEST(ListTest, BasicListOperations) {
    // Test that our list operations work correctly
    
    // Create some test data
    static int values[3] = {10, 20, 30};
    
    // Test list_make1
    List* list1 = list_make1(&values[0]);
    ASSERT_NE(list1, nullptr);
    ASSERT_EQ(list1->length, 1);
    ASSERT_NE(list1->head, nullptr);
    
    // Check we can iterate
    ListCell* lc;
    int count = 0;
    foreach(lc, list1) {
        void* ptr = lfirst(lc);
        std::cout << "List1 item " << count << ": ptr=" << ptr 
                  << " (address of values[0]=" << &values[0] << ")" << std::endl;
        ASSERT_EQ(ptr, &values[0]);
        count++;
    }
    ASSERT_EQ(count, 1);
    
    // Test lappend
    List* list2 = NIL;
    for (int i = 0; i < 3; i++) {
        list2 = lappend(list2, &values[i]);
    }
    ASSERT_NE(list2, nullptr);
    ASSERT_EQ(list2->length, 3);
    
    // Check iteration
    count = 0;
    foreach(lc, list2) {
        void* ptr = lfirst(lc);
        std::cout << "List2 item " << count << ": ptr=" << ptr 
                  << " (expected=" << &values[count] << ")" << std::endl;
        ASSERT_EQ(ptr, &values[count]);
        count++;
    }
    ASSERT_EQ(count, 3);
    
    // Test list_make2
    List* list3 = list_make2(&values[0], &values[1]);
    ASSERT_NE(list3, nullptr);
    ASSERT_EQ(list3->length, 2);
    
    count = 0;
    foreach(lc, list3) {
        void* ptr = lfirst(lc);
        std::cout << "List3 item " << count << ": ptr=" << ptr 
                  << " (expected=" << &values[count] << ")" << std::endl;
        ASSERT_EQ(ptr, &values[count]);
        count++;
    }
    ASSERT_EQ(count, 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}