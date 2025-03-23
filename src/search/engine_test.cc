#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "search/search.h"

namespace lczero {
namespace {

class MockSearchFactory : public SearchFactory {
 public:
  MOCK_METHOD(std::string_view, GetName, (), (const, override));
  MOCK_METHOD(void, PopulateParams, (OptionsParser*), (const, override));
  MOCK_METHOD(std::unique_ptr<SearchBase>, CreateSearch,
              (UciResponder*, const OptionsDict*), (const, override));
};

class EngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code if needed
  }

  void TearDown() override {
    // Cleanup code if needed
  }

  MockSearchFactory mock_search_factory_;
};

TEST_F(EngineTest, TestSearchFactoryName) {
  EXPECT_CALL(mock_search_factory_, GetName())
      .WillOnce(::testing::Return("MockSearch"));

  EXPECT_EQ(mock_search_factory_.GetName(), "MockSearch");
}

}  // namespace
}  // namespace lczero
