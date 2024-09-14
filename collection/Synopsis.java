import java.util.*;
import java.io.*;
/**
 * Processing the synposis section and the like
 */
public class Synopsis {
  /**
   * Find the pagetitle appearing in the tag <title>
   * @param	theList	the lines of the papge as a List
   * @return	the title, empty if none found
   */
  public static String findPageTitle( List<String> theList ) { 
    int p, q;
    // initialize the titleString with ""
    String titleString = "";
    for ( String w : theList ) {
      // the title must appear between <title> and </title>
      // </title> may appear in the following line...
      // can terminate the loop as soon as the string found
      if ( ( p = w.indexOf( "<title>" ) ) >= 0 ) {
        if ( ( q = w.indexOf( "</title>", p + 7 ) ) < 0 ) {
          q = w.length();
        }
        titleString = w.substring( p + 7, q );
        break;
      }
    }
    return titleString;
  }

  /**
   * Find the synopsis
   * @param	theList	the lines of the papge as a List
   * @return	the synopsis, empty if none found
   */
  public static String findSynopsis( List<String> theList ) {
    String synopsisString = "";
    boolean inSynopsis = false;
    for ( String w : theList ) {
      if ( !inSynopsis ) inSynopsis = startSynopsis( w );
      else if ( !w.startsWith( "==" ) ) synopsisString += w;
      else break;
    }
    synopsisString = cleanBrace( synopsisString );
    return synopsisString;
  }
  /**
   * check if the string is the start of a synopsis
   * @param	w	the string
   */
  private static boolean startSynopsis( String w ) {
    return w.startsWith( "==" ) &&
            ( w.indexOf( "あらすじ" ) >= 0 ||
              w.indexOf( "ストーリー" ) >= 0 ||
              w.indexOf( "物語" ) >= 0 ||
              w.indexOf( "ものがたり" ) >= 0 ||
              w.indexOf( "内容" ) >= 0 ||
              w.indexOf( "概略" ) >= 0 ||
              w.indexOf( "概説" ) >= 0 ||
              w.indexOf( "概説" ) >= 0 ||
              w.indexOf( "番組内容" ) >= 0 
            );
  }

  /**
   * find the page title and synopsis
   * 
   * @param	theList	the input list of string data
   * @return	a two-element array containing the two strings
   */
  public static String[] findPageTitleAndSynopsis(
      List<String> theList ) { 
    String[] result = new String[2];
    result[ 0 ] = findPageTitle( theList );
    result[ 1 ] = findSynopsis( theList );
    return result;
  }

  /**
   * remove all the brackets appearing in a string
   * where the cross references can be ignored
   * double brackets may inside another set of brackets
   *
   * @param	w	the input string
   * @return	the string after removal
   */
  public static String cleanBracket( String w ) {
    StringBuilder z = new StringBuilder();
    String x, w2, sub;
    int p, q, r = 0, depth, maxDepth;
    while ( true ) {
      //----- p is the position of the next exterior brackets
      p = w.indexOf( "[[", r );
      //----- if no more open brackets, terminate the loop
      if ( p < 0 ) {
        z.append( w.substring( r ) );
        break;
      }
      //----- otherwise, append the part before the brackets
      z.append( w.substring( r, p ) );
      //----- use r to search for the matching close brackets
      depth = 1;
      maxDepth = 1;
      r = p + 2;
      while ( r < w.length() && depth != 0 ) {
        if ( w.charAt( r ) == '[' && w.charAt( r + 1 ) == '[' ) {
          depth += 1;
          maxDepth = Math.max( maxDepth, depth );
          r += 2;
        }
        else if ( w.charAt( r ) == ']' && w.charAt( r + 1 ) == ']' ) {
          depth -= 1;
          r += 2;
        }
        else {
          r += 1;
        }
      }
      //----- obtain the substring between the two markers
      sub = w.substring( p + 2, r );
      if ( sub.endsWith( "]]" ) ) {
        sub = sub.substring( 0, sub.length() - 2 );
      }
      // System.out.printf( "%d, %d, %s%n", p, r, maxDepthNot1 );
      //----- if the specification is not file
      //----- and the max depth is 1, extract the string after |
      //----- if there is no |, use the entire substring
      if ( maxDepth == 1 &&
           !sub.startsWith( "File" ) &&
           !sub.startsWith( "file" ) ) {
        q = sub.lastIndexOf( "|" );
        if ( q >= 0 ) sub = sub.substring( q + 1 );
        z.append( sub );
      }
    }
    //----- here single brackets will be removed
    w2 = z.toString();
    z = new StringBuilder();
    r = 0;
    while ( true ) {
      p = w2.indexOf( "[", r );
      if ( p < 0 ) {
        z.append( w2.substring( r ) );
        break;
      }
      if ( ( q = w2.indexOf( "]", p ) ) >= 0 ) {
        r = q + 1;
      }
      else {
        break;
      }
    }
    return z.toString();
  }

  /**
   * clean out the sections surrounded by the braces
   * the cleaning is line by line
   * @param	list	some list of strings
   * @return	ArrayList of Strings after cleaning
   */
  public static ArrayList<String> cleanBrace(
      List<String> list ) {
    ArrayList<String> cleaned = new ArrayList<String>();
    for ( String w : list ) {
      cleaned.add( cleanBrace( w ) );
    }
    return cleaned;
  }
  /**
   * clean out le gt
   * @param	some string
   * @return	string after cleaning
   */
  public static String cleanLtGt( String w ) {
    StringBuilder z = new StringBuilder( w );
    int p, q;
    while ( true ) {
      p = z.indexOf( "&lt;" );
      if ( p < 0 ) break;
      q = z.indexOf( "&gt;", p );
      if ( q < 0 ) break;
      z.delete( p, q + 4 );
    }
    return z.toString();
  }

  /**
   * clean out the sections surrounded by the braces
   * @param	some string
   * @return	string after cleaning
   */
  public static String cleanBrace( String w ) {
    StringBuilder z = new StringBuilder( cleanLtGt( w ) );
    String x, x2;
    int p, q, r;
    ArrayList<String> listA = new ArrayList<String>();
    ArrayList<String> listB = new ArrayList<String>();
    ArrayList<String> listC = new ArrayList<String>();
    /* **************************************************
     * first split the text into the parts not including
     * {{}} and those that do not
     * ************************************************** */
    r = 0;
    while ( true ) {
      p = z.indexOf( "{{", r );
      if ( p < 0 ) {
        listA.add( z.substring( r ) );
        listB.add( "" );
        break;
      }
      q = z.indexOf( "}}", p + 2 );
      if ( q >= 0 ) {
        listA.add( z.substring( r, p ) );
        listB.add( z.substring( p, q + 2 ) );
        r = q + 2;
      }
      else {
        listA.add( z.substring( r ) );
        listB.add( "" );
        break;
      }
    }
    /* **************************************************
     * Next, process the parts containing {{}}
     * process each element of listB and store in listC
     * ************************************************** */
    for ( int i = 0; i < listB.size(); i ++ ) {
      x = listB.get( i );
      /* ************************************************
       * if the string is empty, no processing is necessary
       * ************************************************ */
      if ( x.length() == 0 ) {
        listC.add( x );
        continue;
      }
      //----- x2 <- the part without {{}}
      x2 = x.toLowerCase();
      if ( x.endsWith( "}}" ) ) {
        x2 = x2.substring( 2, x2.length() - 2 );
      }
      else {
        x2 = x2.substring( 2, x2.length() );
      }
      if ( x2.startsWith( "基礎情報" ) ) {
        listC.add( x );
        continue;
      }
      else if (
          x2.startsWith( "enlink|" ) || x2.startsWith( "harvnb" ) ||
          x2.startsWith( "interlang" ) || x2.startsWith( "lang|" ) ||
          x2.startsWith( "青空" ) || x2.startsWith( "要出典" ) ) {
        p = x.lastIndexOf( "|" );
        x = x.substring( p + 1, x.length() - 2 );
      }
      else if ( x2.length() == 1 ) {
        x = x2;
      }
      else if ( x2.startsWith( "仮リンク|" ) ||
                x2.startsWith( "読み仮名|" ) || x2.startsWith( "en|" ) ||
                x2.startsWith( "notice|" ) || x2.startsWith( "ruby|" )) {
        p = x.indexOf( "|" );
        q = x.indexOf( "|", p + 1 );
        if ( q < 0 ) {
          q = x.length() - 2;
        }
        x = x.substring( p + 1, q );
      }
      else if ( x2.startsWith( "en|" ) || x2.startsWith( "indent|" ) ||
           x2.startsWith( "interp|" ) || x2.startsWith( "lang-" ) ||
           x2.startsWith( "small|" ) || x2.startsWith( "補助漢字" )) {
        p = x.indexOf( "|" );
        x = x.substring( p, x.length() - 2 );
      }
      else if (
           x2.startsWith( "book" ) || x2.startsWith( "cit" ) ||
           x2.startsWith( "clear" ) || x2.startsWith( "cquot" ) ||
           x2.startsWith( "efn" ) || x2.startsWith( "empty" ) ||
           x2.startsWith( "iso" ) || x2.startsWith( "main" ) ||
           x2.startsWith( "ndljp" ) || x2.startsWith( "pdf" ) ||
           x2.startsWith( "quot" ) || x2.startsWith( "r|" ) ||
           x2.startsWith( "refn|" ) || x2.startsWith( "refnest|" ) ||
           x2.startsWith( "sectstub" ) || x2.startsWith( "see " ) ||
           x2.startsWith( "sfn" ) || x2.startsWith( "tv-" ) ||
           x2.startsWith( "twitter" ) || x2.startsWith( "webarch" ) ||
           x2.startsWith( "youtube|" ) || x2.startsWith( "節" ) ||
           x2.startsWith( "前後番組" ) || x2.startsWith( "誰" ) ||
           x2.startsWith( "注意" ) || x2.startsWith( "不十分なあらすじ" ) ||
           x2.startsWith( "放送ライブラリー" ) ||
           x2.startsWith( "要あらすじ" ) ) {
        x = "";
      }
      else if ( x.indexOf( "|date=" ) >= 0 ) {
        x = "";
      }

      listC.add( x );
    }
    z = new StringBuilder();
    // build back string using listA and listB
    String y;
    for ( int i = 0; i < listC.size(); i ++ ) {
      y = cleanBracket( listA.get( i ) );
      z.append( y );
      z.append( listC.get( i ) );
    }
    return z.toString();
  }

  private static String testString = "" +
	"[[屋久島]]に住むヒロイン・日高満天（まんてん）が" +
	"[[鹿児島市|鹿児島]]でバスガイドになるために島を出るところから物語はスタート。" +
	"物語の舞台は[[大阪]]などに移り、次は[[気象予報士]]になるための勉強を開始。" +
	"そんな中で漁船の遭難で行方不明になっていた父と再会するなどの出来事を通し、最終的には満天は[[宇宙飛行士]]となり、宇宙からの[[天気予報]]を伝えるようになる。" +
	"「電子・光学などのセンサー全盛の時代に宇宙から人間が天気予報をすること」にどれだけの意味・値打ちがあるのかと作中でも疑問が提示されるが、満天が自分の言葉で惑星・地球の美しさを伝えようとしたこころが、日本の子供たちに確実に根付いていることを感じさせて、物語は終了する。" +
	"[[屋久島]]に住むヒロイン・日高満天（まんてん）が[[鹿児島市|鹿児島]]でバスガイドになるために島を出るところから物語はスタート。" +
	"物語の舞台は[[大阪]]などに移り、次は[[気象予報士]]になるための勉強を開始。" +
	"そんな中で漁船の遭難で行方不明になっていた父と再会するなどの出来事を通し、最終的には満天は[[宇宙飛行士]]となり、宇宙からの[[天気予報]]を伝えるようになる。" +
	"「電子・光学などのセンサー全盛の時代に宇宙から人間が天気予報をすること」にどれだけの意味・値打ちがあるのかと作中でも疑問が提示されるが、満天が自分の言葉で惑星・地球の美しさを伝えようとしたこころが、日本の子供たちに確実に根付いていることを感じさせて、物語は終了する。" +
	"[[屋久島]]に住むヒロイン・日高満天（まんてん）が[[鹿児島市|鹿児島]]でバスガイド>になるために島を出るところから物語はスタート。" +
	"物語の舞台は[[大阪]]などに移り、次は[[気象予報士]]になるための勉強を開始。" +
	"そんな中で漁船の遭難で行方不明になっていた父と再会するなどの出来事を通し、最終的" +
	"には満天は[[宇宙飛行士]]となり、宇宙からの[[天気予報]]を伝えるようになる。" +
	"「電子・光学などのセンサー全盛の時代に宇宙から人間が天気予報をすること」にどれだ" +
	"けの意味・値打ちがあるのかと作中でも疑問が提示されるが、満天が自分の言葉で惑星・" +
	"地球の美しさを伝えようとしたこころが、日本の子供たちに確実に根付いていることを感" +
	"じさせて、物語は終了する。" +
        "";

  /**
   * This main is only for testing the functionality.
   */
  public static void main( String[] args ) {
    String w = cleanBracket( testString );
    System.out.println( w );
  }

}
