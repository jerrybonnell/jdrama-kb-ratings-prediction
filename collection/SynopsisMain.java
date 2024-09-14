import java.util.*;
import java.io.*;
/**
 * The source file for extracting synopsis and
 * outputting the result to a file or System.out
 */
public class SynopsisMain {
  /**
   * the main method
   */
  public static void main( String[] args )
      throws FileNotFoundException {

    //---- sc is the scanner for the source file
    File inFile = new File( args[ 0 ] );
    Scanner sc = new Scanner( inFile );

    //---- st is the stream for the output file
    PrintStream st;
    if ( args.length >= 2 ) {
      st = new PrintStream( new File( args[ 1 ] ) );
    }
    else {
      st = System.out;
    }

    String w;
    LinkedList<String> list = new LinkedList<String>();
    ArrayList<String[]> data = new ArrayList<String[]>();
    String[] result;

    /* **********************************************
     * Accumulate the lines between <page> and </page>
     * in a linkedlist of string.
     *
     * Process the list with the findPageTileAndSynposis
     * method of the class Synopsis.
     *
     * The method results a string array of two elements.
     *
     * Add the array to the data.
     *
     * When complete,
     *     output the data with a tab in between.
     *
     * ********************************************** */
    while ( sc.hasNext() ) {
      w = sc.nextLine();
      if ( w.indexOf( "<page>" ) >= 0 ) {
        list = new LinkedList<String>();
      }
      list.add( w );
      if ( w.indexOf( "</page>" ) >= 0 ) {
        System.out.println( list.get( 1 ) );
        result = Synopsis.findPageTitleAndSynopsis( list );
        data.add( result );
      }
    }
    for ( String[] r : data ) {
      st.printf( "%s\t%s%n", r[0], r[1] );
    }
    if ( args.length >= 2 ) {
      st.flush();
      st.close();
    }
  }
}
