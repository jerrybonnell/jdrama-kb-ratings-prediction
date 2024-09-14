/**
 * Compute similarity between two strings using DP
 */
public class DPDistance {

  /* *************************************************
   * constants, there will be four values
   * ************************************************* */

  public static final int AVE_POS = 0;
  public static final int A_POS = 1;
  public static final int B_POS = 2;
  public static final int MAX_POS = 3;

  /**
   * @return	the four values in a double array
   *		similarity / the average length
   *		similarity / the length of a
   *		similarity / the length of b
   *		the larger of the previous two
   * @param	a	the string number 1
   * @param	b	the string number 2
   */
  public static double[] similarityRatio( String a, String b ) {
    double s = similarity( a, b );
    double[] values = new double[4];
    values[0] = 2 * s / (a.length() + b.length() );
    values[1] = s / a.length();
    values[2] = s / b.length();
    values[3] = Math.max( values[1], values[2] );
    return values;
  }

  /**
   * Use dynamic programming to compute similarity
   * @param	a	the string number 1
   * @param	b	the string number 2
   * @return	similarity value as an int
   */
  public static int similarity( String a, String b ) {
    char[] x = a.toCharArray();
    char[] y = b.toCharArray();

    //---- Table initialization
    int[][] table = new int[ x.length ][ y.length ];

    //---- Handle boundary cases
    for ( int i = 0; i < x.length; i ++ )
      table[ i ][ 0 ] = ( x[ i ] == y[ 0 ] ) ? 1: 0;
    for ( int j = 0; j < y.length; j ++ )
      table[ 0 ][ j ] = ( x[ 0 ] == y[ j ] ) ? 1: 0;

    //---- Compute other elements
    for ( int i = 1; i < x.length; i ++ ) {
      for ( int j = 1; j < y.length; j ++ ) {
        if ( x[i] == y[j] ) table[i][j] = table[i-1][j-1] + 1;
        else table[i][j] = Math.max( table[i][j-1], table[i-1][j] );
      }
    }

    // The last entry is the value to return
    return table[ x.length - 1 ][ y.length - 1 ];
  }

  /**
   * The main is for testing the functionality
   */
  public static void main( String[] args ) {
    double[] x;
    String a, b;
    a = "私の　大切な人！";
    b = "私の 大切な人!";
    x = similarityRatio(a, b);
    System.out.printf( "%.3f,%.3f,%.3f,%.3f%n", x[0], x[1], x[2], x[3] );
    a = "私の大切な人";
    b = "私の大切な人";
    x = similarityRatio(a, b);
    System.out.printf( "%.3f,%.3f,%.3f,%.3f%n", x[0], x[1], x[2], x[3] );
    a = "ハガネの女season2";
    b = "孤独のグルメSeason5";
    x = similarityRatio(a, b);
    System.out.printf( "%.3f,%.3f,%.3f,%.3f%n", x[0], x[1], x[2], x[3] );
    a = "99.9−刑事専門弁護士−";
    b = "99.9-刑事専門弁護士-";
    x = similarityRatio(a, b);
    System.out.printf( "%.3f,%.3f,%.3f,%.3f%n", x[0], x[1], x[2], x[3] );
  }
}
