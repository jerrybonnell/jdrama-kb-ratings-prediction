import java.util.*;
import java.io.*;
/**
 * Class for merging the artv-extracted data and
 * the synopsis data
 */

public class Merge {
  /**
   * obtain the 3rd column in the csv
   * @param	w	the input
   */
  public static String getThirdColumn( String w ) {
    int q, p = -1;
    p = w.indexOf( "\"", p + 1 );
    p = w.indexOf( "\"", p + 1 );
    p = w.indexOf( "\"", p + 1 );
    p = w.indexOf( "\"", p + 1 );
    p = w.indexOf( "\"", p + 1 );
    q = w.indexOf( "\"", p + 1 );
    return w.substring( p + 1, q );
  }

  /**
   * main
   */
  public static void main( String[] args )
      throws FileNotFoundException {
    if ( args[2].equals( args[0] ) || !args[1].endsWith( ".csv" ) ) {
      throw new IllegalArgumentException(
          "file name identical or not csv" );
    }

    /* *************************************************
     * read data from the synopsis file (args[0])
     * ************************************************* */
     
    File synFile = new File( args[0] );
    Scanner synSc = new Scanner( synFile );
    ArrayList<String[]> synData = new ArrayList<String[]>();
    while ( synSc.hasNext() ) {
      synData.add( synSc.nextLine().split("\t") );
    }
    synSc.close();

    /* *************************************************
     * read data from the artv data file (args[1])
     * ************************************************* */
    File arFile = new File( args[1] );
    Scanner arSc = new Scanner( arFile );
    ArrayList<String> arData = new ArrayList<String>();
    String header = arSc.nextLine();
    while ( arSc.hasNext() ) {
      arData.add( arSc.nextLine() );
    }
    arSc.close();

    /* *************************************************
     * obtain the name lists
     * ************************************************* */
    String[] query, master;
    master = new String[ synData.size() ];
    for ( int i = 0; i < synData.size(); i ++ ) {
      master[i] = synData.get( i )[0];
    }
    query = new String[ arData.size() ];
    for ( int i = 0; i < arData.size(); i ++ ) {
      query[i] = getThirdColumn( arData.get( i ) );
    }

    /* *************************************************
     * Execute matching
     * ************************************************* */
    Matching matcher = new Matching( query, master );
    matcher.match();

    /* *************************************************
     * Append the results
     * ************************************************* */
    File outFile = new File( args[2] );
    PrintStream out = new PrintStream( outFile );
    String header2 = header + ",\"synopsis\"," + Matching.HEADER;
    out.println( header2 );

    /* *************************************************
     * Output the results
     * ************************************************* */
    int k;
    for ( int i = 0; i < arData.size(); i ++ ) {
      out.print( arData.get( i ) + "," );
      k = matcher.getMapping(i);
      if ( k < 0 ) out.print( "\"NO_NAME_MATCH\"," );
      else if ( synData.get( k ).length == 1 )
        out.print( "\"NO_SYNOPSIS\"," );
      else out.printf( "\"%s\",", synData.get( k )[1] );
      out.println( matcher.matchSignature( i ) );
    }
    out.flush();
    out.close();
  }
}
