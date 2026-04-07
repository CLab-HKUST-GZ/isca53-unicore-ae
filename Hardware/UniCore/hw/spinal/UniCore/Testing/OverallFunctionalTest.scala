package UniCore.Testing
import scala.language.postfixOps
import UniCore.Testing.TestCase.{Test_SFPMA_4b8b, Test_SFPMA_4b8b16b, Test_CPE_4b8b, Test_CPE_4b8b16b}

object OverallFunctionalTest extends App {

  // **** Functional Test 1 ****
  Test_SFPMA_4b8b.runAll()

  // **** Functional Test 2 ****
  Test_SFPMA_4b8b16b.runAll()

  // **** Functional Test 3 ****
  Test_CPE_4b8b.runAll()

  // **** Functional Test 4 ****
  Test_CPE_4b8b16b.runAll()

}
